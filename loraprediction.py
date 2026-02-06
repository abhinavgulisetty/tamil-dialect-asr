import torch
import os
import glob
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# Configuration
BASE_MODEL = "vasista22/whisper-tamil-medium"
LORA_PATH = "./whisper-tamil-lora"  # Your trained LoRA adapters
AUDIO_FOLDER = "/root/abhinav/Dialect/Test"  # Folder containing .wav files
OUTPUT_FILE = "lora_predictions.txt"
BATCH_SIZE = 16 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("Tamil Speech Recognition with LoRA")
print("="*60)
print(f"Device: {device}")
print(f"LoRA Model: {LORA_PATH}")
print(f"Audio Folder: {AUDIO_FOLDER}")
print(f"Output File: {OUTPUT_FILE}")

# Load processor
processor = WhisperProcessor.from_pretrained(LORA_PATH, language="Tamil", task="transcribe")

# Load base model
print("\nLoading base model...")
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
).to(device)

# Load LoRA adapters
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print("✓ Model loaded successfully\n")

# Setup generation config
gen_config = GenerationConfig.from_pretrained(BASE_MODEL)
gen_config.update(
    forced_decoder_ids=processor.get_decoder_prompt_ids(language="tamil", task="transcribe"),
    language="tamil",
    task="transcribe",
    max_new_tokens=225,
    return_timestamps=False
)

# Find audio files
print("Searching for audio files...")
audio_files = sorted(glob.glob(os.path.join(AUDIO_FOLDER, "*.wav")))

if not audio_files:
    print(f"ERROR: No .wav files found in {AUDIO_FOLDER}")
    exit(1)

print(f"Found {len(audio_files)} audio files\n")

# Initialize output file
print(f"Writing predictions to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("# Tamil Speech Recognition Predictions\n")
    f.write(f"# Total files: {len(audio_files)}\n")
    f.write("# Format: filename | transcription\n")
    f.write("="*80 + "\n\n")

# Process audio files in batches
print(f"Processing {len(audio_files)} files in batches of {BATCH_SIZE}...\n")

for i in tqdm(range(0, len(audio_files), BATCH_SIZE), desc="Transcribing"):
    batch_paths = audio_files[i : i + BATCH_SIZE]
    audio_inputs = []
    
    # Load audio files
    for path in batch_paths:
        try:
            audio_array, _ = librosa.load(path, sr=16000) 
            audio_inputs.append(audio_array)
        except Exception as e:
            print(f"\nWarning: Failed to load {path}: {e}")
            audio_inputs.append(np.zeros(16000))  # Add silence as placeholder
    
    # Process batch
    inputs = processor(audio_inputs, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = inputs.input_features.to(device)
    
    if torch.cuda.is_available():
        input_features = input_features.to(torch.bfloat16)
    
    # Generate transcriptions
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            generation_config=gen_config
        )
    
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Write results with filenames
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for path, text in zip(batch_paths, transcriptions):
            filename = os.path.basename(path)
            f.write(f"{filename} | {text.strip()}\n")

print(f"\n{'='*60}")
print(f"✓ Complete! Predictions saved to: {OUTPUT_FILE}")
print(f"✓ Total files processed: {len(audio_files)}")
print(f"{'='*60}")