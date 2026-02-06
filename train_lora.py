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

# 3. Preprocessing with SpecAugment
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="Tamil", task="transcribe")
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

# SpecAugment for data augmentation (applied during training only)
import random
def apply_spec_augment(input_features, freq_mask_param=15, time_mask_param=35):
    """Apply SpecAugment for robustness (masks time/freq in spectrogram)"""
    # Randomly mask frequency bands
    for _ in range(2):  # Apply 2 frequency masks
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, input_features.shape[0] - f)
        input_features[f0:f0+f, :] = 0
    
    # Randomly mask time steps
    for _ in range(2):  # Apply 2 time masks
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, input_features.shape[1] - t)
        input_features[:, t0:t0+t] = 0
    
    return input_features

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

print(f"Training samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    apply_augmentation: bool = False  # Flag for SpecAugment
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Explicitly cast input_features to bfloat16 to match model weights
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        
        # Apply SpecAugment during training only for robustness
        if self.apply_augmentation:
            batch_size = batch["input_features"].shape[0]
            for i in range(batch_size):
                if torch.rand(1).item() > 0.5:  # 50% probability
                    batch["input_features"][i] = torch.tensor(
                        apply_spec_augment(batch["input_features"][i].numpy()),
                        dtype=torch.bfloat16
                    )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Mask padding tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # Remove BOS token if present at the start
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, apply_augmentation=True)

# 5. Metrics - WER + CER for comprehensive Tamil evaluation
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

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

    # Calculate both WER and CER (CER is better for character-based languages like Tamil)
    wer = 100 * wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
    cer = 100 * cer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
    
    return {"wer": wer, "cer": cer}

# 6. Load Model & Apply LoRA
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False

# Force language/task decoding
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="tamil", task="transcribe")
model.config.suppress_tokens = []

# Apply PEFT/LoRA - SOTA configuration for multi-dialect robustness
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=64,             # Increased rank for 4 dialects (more capacity)
    lora_alpha=128,   # Alpha (2x Rank)
    lora_dropout=0.05, # Lower dropout for better learning
    # Comprehensive module coverage for dialect variations
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# 7. Training Arguments - SOTA configuration for multi-dialect Tamil ASR
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,  # Adjusted for expanded LoRA modules
    gradient_accumulation_steps=2,   # Effective batch = 32
    learning_rate=5e-4,              # Optimal for LoRA with higher rank
    warmup_steps=500,                # Longer warmup for dialect stability
    warmup_ratio=0.05,               # 5% warmup for multi-dialect learning
    num_train_epochs=5,              # 5 epochs * 4100 samples = good coverage
    weight_decay=0.01,               # Regularization for dialect generalization
    max_grad_norm=1.0,               # Gradient clipping for training stability
    label_smoothing_factor=0.1,      # Smoothing for better generalization
    gradient_checkpointing=False,    
    fp16=False,     
    bf16=True,                       
    dataloader_num_workers=8,        # Maximize data throughput
    eval_strategy="steps",       
    per_device_eval_batch_size=16, 
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=1,          
    save_steps=400,                  # Frequent checkpointing
    eval_steps=400,                  # Frequent evaluation
    logging_steps=50,                
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",     # CER is better for Tamil (character-based)
    greater_is_better=False,
    save_total_limit=3,              
    optim="adamw_torch_fused",       # H100 optimized
    lr_scheduler_type="cosine_with_restarts",  # Better than plain cosine
    remove_unused_columns=False,     
    label_names=["labels"],          
    group_by_length=True,            # Critical for mixed-length efficiency
    length_column_name="duration",   # Use duration for grouping
    dataloader_pin_memory=True,      # H100 optimization
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