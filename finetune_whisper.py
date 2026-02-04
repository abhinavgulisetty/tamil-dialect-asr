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

# 1. Configuration
MODEL_ID = "vasista22/whisper-tamil-medium"
TRAIN_JSON = "path/to/train.json"  # Update with actual path
TEST_JSON = "path/to/test.json"    # Update with actual path
OUTPUT_DIR = "./whisper-tamil-finetuned"

# 2. Load Dataset
dataset = load_dataset("json", data_files={"train": TRAIN_JSON, "test": TEST_JSON})

# 3. Preprocessing
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="Tamil", task="transcribe")

# Resample audio to 16kHz (Whisper requirement)
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # Load and resample audio data
    audio = batch["audio_filepath"]

    # Compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Apply preprocessing
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss calculation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Evaluation Metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replacing -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 6. Load Model
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Disable cache during training; re-enable for inference
model.config.use_cache = False
# Set language and task for generation
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 7. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,  # Adjust based on GPU VRAM
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,  # Use fp16 if CUDA is available
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# 8. Initialize Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# 9. Start Training
if __name__ == "__main__":
    trainer.train()
    
    # Save final model and processor
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
