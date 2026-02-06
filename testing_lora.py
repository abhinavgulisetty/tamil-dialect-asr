import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# 1. Setup - LoRA Model
BASE_MODEL = "vasista22/whisper-tamil-medium"
LORA_PATH = "./whisper-tamil-lora"  # Your trained LoRA adapters
TEST_JSON = "test_manifest_clean.json"  # Updated to correct file
BATCH_SIZE = 16 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading LoRA model for evaluation...")

# 2. Load Metrics, Processor, and Model
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")  # CER is better for Tamil

# Tamil text normalization (same as training)
def normalize_tamil(text):
    """Simple normalization for Tamil text"""
    text = " ".join(text.split())
    return text.strip()

# Load processor
processor = WhisperProcessor.from_pretrained(LORA_PATH, language="Tamil", task="transcribe")

# Load base model
model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.bfloat16
).to(device)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print(f"✓ Loaded LoRA model from {LORA_PATH}")

# 3. Load Test Data
dataset = load_dataset("json", data_files={"test": TEST_JSON})["test"]
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

# 4. Setup GenerationConfig
# This bundles the forced_decoder_ids into a format the model expects
gen_config = GenerationConfig.from_pretrained(BASE_MODEL)
gen_config.update(
    forced_decoder_ids=processor.get_decoder_prompt_ids(language="tamil", task="transcribe"),
    language="tamil",
    task="transcribe",
    max_new_tokens=225,
    return_timestamps=False
)

# 5. Evaluation Loop
references = []
predictions = []

print(f"\nEvaluating {len(dataset)} samples with LoRA adapters on H100...")

for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    # Slice the dataset for the current batch
    batch_data = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
    
    # Process audio batch
    audio_arrays = [x["audio_filepath"]["array"] for x in batch_data]
    inputs = processor(audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = inputs.input_features.to(device).to(torch.bfloat16)
    
    # Generate using the config object
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            generation_config=gen_config
        )
    
    # Decode and store
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    predictions.extend(transcriptions)
    references.extend(batch_data["text"])

# 6. Compute Final Metrics with Normalization
# Normalize predictions and references (same as training)
predictions_norm = [normalize_tamil(pred) for pred in predictions]
references_norm = [normalize_tamil(ref) for ref in references]

# Handle empty strings
predictions_norm = [pred if len(pred) > 0 else " " for pred in predictions_norm]
references_norm = [ref if len(ref) > 0 else " " for ref in references_norm]

final_wer = 100 * wer_metric.compute(predictions=predictions_norm, references=references_norm)
final_cer = 100 * cer_metric.compute(predictions=predictions_norm, references=references_norm)

print("\n" + "="*50)
print(f"FINAL TEST WER: {final_wer:.2f}%")
print(f"FINAL TEST CER: {final_cer:.2f}%")
print(f"   Total Samples: {len(predictions)}")
print("="*50)

# 7. Show Comparison Examples
print("\n--- Sample Results ---")
for i in range(min(5, len(predictions))):
    print(f"\n[SAMPLE {i+1}]")
    print(f"REFERENCE: {references[i]}")
    print(f"PREDICTED: {predictions[i]}")
    # Show if they match
    match = "✓ MATCH" if references_norm[i] == predictions_norm[i] else "✗ MISMATCH"
    print(f"Status: {match}")

# 8. Accuracy stats
correct = sum(1 for p, r in zip(predictions_norm, references_norm) if p == r)
accuracy = (correct / len(predictions)) * 100
print(f"\n{'='*50}")
print(f"Exact Match Accuracy: {accuracy:.2f}% ({correct}/{len(predictions)})")
print(f"{'='*50}")
