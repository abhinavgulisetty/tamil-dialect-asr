import argparse
import json
import os
import random
from typing import Dict, List

import torch
from datasets import Audio, Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import WhisperModel, WhisperProcessor


DIALECT_MAP = {
    "Central_Dialect": "Central",
    "Northern_Dialect": "Northern",
    "Southern_Dialect": "Southern",
    "Western_Dialect": "Western",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_manifest(data_root: str) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    for folder_name, dialect in DIALECT_MAP.items():
        dialect_dir = os.path.join(data_root, folder_name)
        if not os.path.isdir(dialect_dir):
            continue

        for root, _, files in os.walk(dialect_dir):
            for filename in files:
                if not filename.lower().endswith(".wav"):
                    continue
                manifest.append(
                    {
                        "audio_filepath": os.path.join(root, filename),
                        "label": dialect,
                    }
                )
    if not manifest:
        raise FileNotFoundError(f"No .wav files found under: {data_root}")
    return manifest


class DialectClassifier(nn.Module):
    def __init__(self, encoder: WhisperModel, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.d_model, num_labels)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        encoder_out = self.encoder.encoder(input_features=input_features)
        pooled = encoder_out.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dialect classifier with Whisper encoder.")
    parser.add_argument("--data-root", default="Dataset", help="Path to dataset root.")
    parser.add_argument("--model-id", default="vasista22/whisper-tamil-medium")
    parser.add_argument("--output-dir", default="dialect_classifier")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-size", type=float, default=0.2)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = build_manifest(args.data_root)
    labels = sorted({item["label"] for item in manifest})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    dataset = Dataset.from_list(manifest)
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(args.model_id, language="Tamil", task="transcribe")

    def preprocess(batch):
        audio = batch["audio_filepath"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["label_id"] = label2id[batch["label"]]
        return batch

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    split = dataset.train_test_split(
        test_size=args.eval_size,
        seed=args.seed,
        stratify_by_column="label_id",
    )
    train_ds = split["train"].with_format("torch")
    eval_ds = split["test"].with_format("torch")

    def collate_fn(features):
        input_features = [f["input_features"] for f in features]
        batch = processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        labels_tensor = torch.tensor([f["label_id"] for f in features], dtype=torch.long)
        return batch["input_features"], labels_tensor

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = WhisperModel.from_pretrained(args.model_id)
    model = DialectClassifier(encoder=encoder, num_labels=len(labels), dropout=args.dropout)
    model.to(device)

    if not args.unfreeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for input_features, labels_tensor in train_loader:
            input_features = input_features.to(device)
            labels_tensor = labels_tensor.to(device)

            optimizer.zero_grad()
            logits = model(input_features)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_features, labels_tensor in eval_loader:
                input_features = input_features.to(device)
                labels_tensor = labels_tensor.to(device)
                logits = model(input_features)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels_tensor).sum().item()
                total += labels_tensor.size(0)

        acc = correct / total if total else 0.0
        avg_loss = train_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} eval_acc={acc:.4f}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "classifier.pt"))

    with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print(f"Saved classifier to {args.output_dir}")


if __name__ == "__main__":
    main()