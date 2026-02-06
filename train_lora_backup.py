import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

# 1. Configuration
MODEL_ID = "vasista22/whisper-tamil-medium"
TRAIN_JSON = "train_manifest_clean.json"
TEST_JSON = "test_manifest_clean.json"
OUTPUT_DIR = "./whisper-tamil-lora"
MAX_LABEL_LENGTH = 448 

# 2. Load Dataset
dataset = load_dataset("json", data_files={"train": TRAIN_JSON, "test": TEST_JSON})

# 3. Preprocessing
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="Tamil", task="transcribe")
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio_filepath"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

# Filter samples that exceed Whisper's token limit
dataset = dataset.filter(lambda x: len(x["labels"]) <= MAX_LABEL_LENGTH)
dataset["train"] = dataset["train"].select(range(50))
dataset["test"] = dataset["test"].select(range(10))
print(f"Training samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Explicitly cast input_features to bfloat16 to match model weights
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Mask padding tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Remove BOS token if present at the start
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Metric - Tamil-friendly (no English normalizer)
metric = evaluate.load("wer")

def normalize_tamil(text):
    """Simple normalization for Tamil text"""
    # Remove extra whitespace and strip
    text = " ".join(text.split())
    return text.strip()

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Apply Tamil-safe normalization (just whitespace cleanup)
    pred_str_norm = [normalize_tamil(pred) for pred in pred_str]
    label_str_norm = [normalize_tamil(label) for label in label_str]

    # Handle empty strings to avoid division by zero errors
    pred_str_norm = [pred if len(pred) > 0 else " " for pred in pred_str_norm]
    label_str_norm = [label if len(label) > 0 else " " for label in label_str_norm]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
    return {"wer": wer}

# 6. Load Model & Apply LoRA
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False

# Force language/task decoding
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="tamil", task="transcribe")
model.config.suppress_tokens = []

# Apply PEFT/LoRA - Expanded for multi-dialect robustness
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=32,             # Rank
    lora_alpha=64,    # Alpha (2x Rank)
    lora_dropout=0.1,
    # Expanded modules for better dialect variation capture
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# 7. Training Arguments - Optimized for 5314 samples with 4 dialects
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,  # Adjusted for expanded LoRA modules
    gradient_accumulation_steps=2,   # Effective batch = 32
    learning_rate=5e-4,              # Slightly lower for stability with more params
    warmup_steps=300,                # ~5% of total steps for larger dataset
    warmup_ratio=0.03,               # Alternative warmup calculation
    num_train_epochs=5,              # 5 epochs * 5314 samples = good coverage
    weight_decay=0.01,               # Regularization for dialect generalization
    gradient_checkpointing=False,    
    fp16=False,     
    bf16=True,                       
    dataloader_num_workers=8,        # Increased for faster data loading
    eval_strategy="steps",       
    per_device_eval_batch_size=16, 
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=1,          
    save_steps=500,                  # More frequent saves (every ~100 samples)
    eval_steps=500,                  # More frequent evaluation
    logging_steps=50,                # Track progress more frequently
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=3,              # Keep best 3 checkpoints to save disk space
    optim="adamw_torch_fused",       # H100 optimized
    lr_scheduler_type="cosine",      # Smooth LR decay for better convergence
    remove_unused_columns=False,     # Required for PEFT
    label_names=["labels"],          # Required for PEFT
    group_by_length=True,            # Optimize batching by audio length
)

# 8. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# 9. Start Training
if __name__ == "__main__":
    trainer.train()
    
    # Save the adapter
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapters saved to {OUTPUT_DIR}")