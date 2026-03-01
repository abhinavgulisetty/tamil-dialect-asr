import argparse
import json
import random
import os


def split_manifest(manifest_path: str, train_path: str, test_path: str, test_ratio: float, seed: int) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    random.seed(seed)
    random.shuffle(lines)

    num_test = int(len(lines) * test_ratio)
    test_lines = lines[:num_test]
    train_lines = lines[num_test:]

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(test_path, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print(
        f"Split {len(lines)} samples into "
        f"{len(train_lines)} train and {len(test_lines)} test entries."
    )
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a JSON manifest into train and test sets for LoRA ASR training."
    )
    parser.add_argument(
        "--manifest",
        default="dataset_manifest.json",
        help="Input manifest file produced by dataset_to_json.py (default: dataset_manifest.json).",
    )
    parser.add_argument("--train-out", default="train_manifest_clean.json", help="Output train manifest.")
    parser.add_argument("--test-out", default="test_manifest_clean.json", help="Output test manifest.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction reserved for test (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest file not found: {args.manifest}")

    split_manifest(args.manifest, args.train_out, args.test_out, args.test_ratio, args.seed)

