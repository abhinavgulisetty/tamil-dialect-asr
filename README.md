# DLRG@DravidianLangTech 2026: Tamil Dialect Speech Recognition and Classification

This repository contains the system code for the DLRG team's submission to the shared task on **Dialect Based Speech Recognition and Classification in Tamil** at **DravidianLangTech@ACL 2026**.

We address both subtasks using a single pre-trained foundation model — [`vasista22/whisper-tamil-medium`](https://huggingface.co/vasista22/whisper-tamil-medium) — adapted through two different fine-tuning strategies.

| Subtask | Approach | Result |
|---|---|---|
| Subtask 1 — Dialect Identification | Full encoder fine-tuning + mean pooling | **73.4% accuracy** |
| Subtask 2 — Dialectal ASR | LoRA (rank 64) + SpecAugment | **0.55 WER** |

---

## Repository Structure

```
tamil-dialect-asr/
├── dataset_to_json.py          # Step 1: build JSON manifest from raw Dataset/ folder
├── prepare_asr_data.py         # Step 2: split manifest into train/test (for ASR pipeline)
├── train_dialect_classifier.py # Subtask 1 — train Whisper encoder classifier
├── train_lora.py               # Subtask 2 — fine-tune Whisper with LoRA for ASR
├── infer_dialect_classifier.py # Subtask 1 — run inference and generate submission file
├── infer_lora_asr.py           # Subtask 2 — run inference and generate submission file
├── merge_lora.py               # (Optional) merge LoRA adapters into base model weights
└── requirements.txt            # Python dependencies
```

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch (adjust for your CUDA version — see requirements.txt header)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset Structure

Place the official Tamil Dialect Speech Dataset in a `Dataset/` folder with the following layout:

```
Dataset/
├── Central_Dialect/
│   ├── SP1_CEN_Text.txt
│   ├── SP1_CEN_audio/
│   │   ├── SP1_CEN_001.wav
│   │   └── ...
│   └── ...
├── Northern_Dialect/
├── Southern_Dialect/
└── Western_Dialect/
```

Test audio files should be placed in a `Test/` folder.

---

## Subtask 1: Dialect Classification

### Training

The classifier is trained end-to-end from the `Dataset/` directory. No preprocessing step is required.

```bash
python train_dialect_classifier.py \
    --data-root Dataset \
    --model-id vasista22/whisper-tamil-medium \
    --output-dir dialect_classifier \
    --epochs 5 \
    --lr 1e-5 \
    --batch-size 8 \
    --unfreeze-encoder
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data-root` | `Dataset` | Path to the dataset root directory |
| `--model-id` | `vasista22/whisper-tamil-medium` | HuggingFace model ID |
| `--output-dir` | `dialect_classifier` | Where to save `classifier.pt` and `label_map.json` |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--batch-size` | `16` | Batch size |
| `--dropout` | `0.2` | Dropout rate on the classifier head |
| `--unfreeze-encoder` | *(flag)* | **Required** for full fine-tuning (critical for dialect discrimination) |
| `--eval-size` | `0.2` | Fraction of data held out for validation |
| `--seed` | `42` | Random seed |

> **Important:** Always pass `--unfreeze-encoder`. Freezing the encoder yields only ~52% accuracy; full fine-tuning reaches 73.4% test accuracy.

### Inference

```bash
python infer_dialect_classifier.py \
    --audio-dir Test \
    --classifier-dir dialect_classifier \
    --output-file abhinavg_Classification_Run1.txt
```

The output file contains one `<file_id> <dialect_label>` entry per line, ready for submission.

---

## Subtask 2: Dialectal ASR (LoRA)

### Step 1 — Build the JSON manifest

```bash
python dataset_to_json.py --data-root Dataset --output dataset_manifest.json
```

### Step 2 — Split into train / test

```bash
python prepare_asr_data.py \
    --manifest dataset_manifest.json \
    --train-out train_manifest_clean.json \
    --test-out test_manifest_clean.json \
    --test-ratio 0.2 \
    --seed 42
```

### Step 3 — Train the LoRA model

```bash
python train_lora.py
```

Training reads `train_manifest_clean.json` and `test_manifest_clean.json` from the working directory and saves LoRA adapters to `./whisper-tamil-lora/`.

**LoRA configuration used in our submission:**

| Parameter | Value |
|---|---|
| Base model | `vasista22/whisper-tamil-medium` |
| LoRA rank (`r`) | 64 |
| LoRA alpha (`α`) | 128 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` |
| Trainable parameters | ~69.2M (~8.3% of total) |
| Effective batch size | 32 (batch 16 × grad accum 2) |
| Peak learning rate | 5×10⁻⁴ (cosine w/ restarts) |
| Epochs | 5 |
| Precision | BF16 |
| SpecAugment | Freq ≤15 bins, Time ≤35 frames |

### Step 4 — Inference

```bash
python infer_lora_asr.py \
    --audio-dir Test \
    --lora-path ./whisper-tamil-lora \
    --output-file abhinavg_Recognition_Run1.txt
```

The output file contains one `<file_id> <transcription>` entry per line, ready for submission.

### (Optional) Merge LoRA weights

To produce a single self-contained model (no PEFT dependency at inference time):

```bash
python merge_lora.py
```

Saves the merged model to `./whisper-tamil-merged/`.

---

## Citation

If you use this code, please cite our system paper:

```
@inproceedings{abhinav2026dlrg,
  title={{DLRG}@{D}ravidian{L}ang{T}ech 2026: Dual-Purpose Whisper Adaptation for Tamil Dialect Identification and Dialectal Speech Recognition},
  author={Gulisetty Abhinav and Tanisha Nanda and Rajalakshmi, Ratnavel and Rameshkannan R},
  booktitle={Proceedings of the Sixth Workshop on Speech, Vision, and Language Technologies for Dravidian Languages},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

---

## Acknowledgements

We thank the organizers of the DravidianLangTech@ACL 2026 shared task for providing the Tamil Dialect Speech Dataset and evaluation framework.
