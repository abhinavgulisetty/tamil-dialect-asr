import argparse
import json
import os
from typing import List, Tuple

import torch
import librosa
from torch.utils.data import DataLoader
from transformers import WhisperModel, WhisperProcessor

from train_dialect_classifier import DIALECT_MAP, DialectClassifier


def list_audio_files(audio_dir: str) -> List[str]:
    audio_files: List[str] = []
    for root, _, files in os.walk(audio_dir):
        for filename in files:
            if filename.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, filename))
    return sorted(audio_files)


def build_output_lines(
    audio_paths: List[str],
    model: DialectClassifier,
    processor: WhisperProcessor,
    id2label: dict,
    batch_size: int,
    device: torch.device,
) -> List[Tuple[str, str]]:
    def collate_fn(paths: List[str]):
        audio_arrays = []
        for path in paths:
            audio_array, _ = librosa.load(path, sr=16000)
            audio_arrays.append(audio_array)

        features = [
            processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
            for audio in audio_arrays
        ]
        batch = processor.feature_extractor.pad(
            {"input_features": features}, return_tensors="pt"
        )
        return paths, batch["input_features"]

    loader = DataLoader(audio_paths, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    results: List[Tuple[str, str]] = []

    model.eval()
    with torch.no_grad():
        for paths, input_features in loader:
            input_features = input_features.to(device)
            logits = model(input_features)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            for path, pred_id in zip(paths, preds):
                file_id = os.path.splitext(os.path.basename(path))[0]
                label = id2label[str(pred_id)]
                results.append((file_id, label))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dialect classifier inference.")
    parser.add_argument("--audio-dir", default="Test", help="Directory with test .wav files.")
    parser.add_argument("--model-id", default="vasista22/whisper-tamil-medium")
    parser.add_argument("--classifier-dir", default="dialect_classifier")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-file", default="abhinavg_Classification_Run1.txt")
    args = parser.parse_args()

    audio_files = list_audio_files(args.audio_dir)
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found under: {args.audio_dir}")

    label_map_path = os.path.join(args.classifier_dir, "label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id2label = label_map["id2label"]

    processor = WhisperProcessor.from_pretrained(args.model_id, language="Tamil", task="transcribe")
    encoder = WhisperModel.from_pretrained(args.model_id)
    model = DialectClassifier(encoder=encoder, num_labels=len(id2label), dropout=0.0)

    classifier_path = os.path.join(args.classifier_dir, "classifier.pt")
    state_dict = torch.load(classifier_path, map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_lines = build_output_lines(
        audio_files, model, processor, id2label, args.batch_size, device
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for file_id, label in output_lines:
            f.write(f"{file_id} {label}\n")

    print(f"Saved classification file to {args.output_file}")


if __name__ == "__main__":
    main()
