import torch
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Configuration
PEFT_MODEL_ID = "./whisper-tamil-lora" # Where you saved the adapters
BASE_MODEL_ID = "vasista22/whisper-tamil-medium"
MERGED_MODEL_DIR = "./whisper-tamil-merged"

print("Loading base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.float16, # Use float16 for merging
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_MODEL_DIR}...")
model.save_pretrained(MERGED_MODEL_DIR)

# Don't forget to save the processor/tokenizer as well
processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language="Tamil", task="transcribe")
processor.save_pretrained(MERGED_MODEL_DIR)

print("Done! You can now load the model directly from 'whisper-tamil-merged'")