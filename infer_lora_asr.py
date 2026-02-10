import argparse
import os
from typing import List

import librosa
import torch
from peft import PeftConfig, PeftModel
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor


def list_audio_files(audio_dir: str) -> List[str]:
    audio_files: List[str] = []
    for root, _, files in os.walk(audio_dir):
        for filename in files:
            if filename.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, filename))
    return sorted(audio_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA Whisper inference for submission.")
    parser.add_argument("--audio-dir", default="Test", help="Directory with test .wav files.")
    parser.add_argument("--lora-path", default="./whisper-tamil-lora")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--output-file", default="abhinavg_Recognition_Run1.txt")
    args = parser.parse_args()

    audio_files = list_audio_files(args.audio_dir)
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found under: {args.audio_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    peft_config = PeftConfig.from_pretrained(args.lora_path)
    base_model_id = peft_config.base_model_name_or_path

    processor = WhisperProcessor.from_pretrained(args.lora_path, language="Tamil", task="transcribe")

    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    gen_config = GenerationConfig.from_pretrained(base_model_id)
    gen_config.update(
        forced_decoder_ids=processor.get_decoder_prompt_ids(language="tamil", task="transcribe"),
        language="tamil",
        task="transcribe",
        max_new_tokens=args.max_new_tokens,
        return_timestamps=False,
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for i in range(0, len(audio_files), args.batch_size):
            batch_paths = audio_files[i : i + args.batch_size]
            audio_arrays = [librosa.load(path, sr=16000)[0] for path in batch_paths]

            inputs = processor(
                audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_features = inputs.input_features.to(device)

            if torch.cuda.is_available():
                input_features = input_features.to(torch.bfloat16)

            with torch.no_grad():
                generated_ids = model.generate(input_features, generation_config=gen_config)

            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for path, text in zip(batch_paths, transcriptions):
                file_id = os.path.splitext(os.path.basename(path))[0]
                f.write(f"{file_id} {text.strip()}\n")

    print(f"Saved recognition file to {args.output_file}")


if __name__ == "__main__":
    main()
