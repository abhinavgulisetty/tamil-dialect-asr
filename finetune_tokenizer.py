import os
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
from huggingface_hub import hf_hub_download

# 1. Configuration
# The Hugging Face repo ID
MODEL_REPO_ID = "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large"
# The specific filename inside the repo (usually model.nemo)
MODEL_FILENAME = "model.nemo" 

# Make sure these files exist (run the cleaning scripts first)
TRAIN_MANIFEST = "train_manifest_clean.json"
TEST_MANIFEST = "test_manifest_clean.json"

# H100 Specifics
# H100 has 80GB memory. Batch size 64 is safe, 128 is aggressive/optimal.
BATCH_SIZE = 64  
# Utilize TransformerEngine / BF16 for maximum speed on H100
PRECISION = "bf16-mixed" 

def main():
    print(f"--- Starting H100 Finetuning Preparation ---")

    # 2. Load Model
    # NeMo's .from_pretrained() only lists NVIDIA NGC models.
    # We must manually download the .nemo file from Hugging Face first.
    print(f"Downloading {MODEL_FILENAME} from {MODEL_REPO_ID}...")
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        print(f"Model downloaded to: {model_path}")
        
        print("Restoring model from local .nemo file...")
        asr_model = EncDecHybridRNNTCTCModel.restore_from(restore_path=model_path)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not download/load model: {e}")
        print(f"Please verify the filename '{MODEL_FILENAME}' exists in the HF repo.")
        return

    # 3. Setup Data Loaders
    print(f"Setting up data config with batch size {BATCH_SIZE}...")
    
    cfg = OmegaConf.create({
        'manifest_filepath': TRAIN_MANIFEST,
        'sample_rate': 16000,
        'batch_size': BATCH_SIZE, 
        'shuffle': True, 
        'num_workers': 16, # High worker count (16-32) prevents H100 from starving
        'pin_memory': True,
        'use_start_end_token': False,
        'trim_silence': True,
        'max_duration': 20.0, 
        'min_duration': 0.5 
    })

    # Setup Training Data
    try:
        asr_model.setup_training_data(train_data_config=cfg)
    except Exception as e:
        print(f"[ERROR] Failed to setup training data. Check if '{TRAIN_MANIFEST}' exists and isn't empty.")
        raise e

    # Setup Validation Data
    test_cfg = cfg.copy()
    test_cfg['manifest_filepath'] = TEST_MANIFEST
    test_cfg['shuffle'] = False
    asr_model.setup_validation_data(val_data_config=test_cfg)

    # 4. Dialect Adaptation Configuration
    # We alter the internal config to lower Learning Rate (LR) and increase Augmentation
    print("Applying Dialect Adaptation settings (Low LR + High SpecAugment)...")
    with open_dict(asr_model.cfg):
        # OPTIMIZER: 
        # Use a lower LR because we are "adapting" to a dialect, not training from scratch.
        # High LR will destroy the base Tamil knowledge.
        asr_model.cfg.optim.lr = 1e-5 
        asr_model.cfg.optim.sched.min_lr = 1e-6
        
        # SPEC AUGMENT: 
        # Increase masking to force the model to look at context rather than memorizing dialect noise
        if 'spec_augment' in asr_model.cfg:
            asr_model.cfg.spec_augment.freq_masks = 2
            asr_model.cfg.spec_augment.time_masks = 5
            asr_model.cfg.spec_augment.freq_width = 27
            asr_model.cfg.spec_augment.time_width = 0.05

    # Apply optimization settings
    asr_model.setup_optimization(optim_config=asr_model.cfg.optim)

    # 5. Trainer Setup for H100
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=50, # 50-100 epochs is usually sufficient for 5k samples
        precision=PRECISION, 
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        logger=False, # Set to True if you want Tensorboard logging
        log_every_n_steps=5,
        val_check_interval=0.5, # Validate twice per epoch to catch overfitting early
        gradient_clip_val=1.0
    )

    # 6. Run Training
    print("Starting Training Loop...")
    trainer.fit(asr_model)

    # 7. Save
    save_path = "indicconformer_tamil_dialect_finetuned.nemo"
    asr_model.save_to(save_path)
    print(f"SUCCESS: Model saved to {save_path}")

if __name__ == "__main__":
    main()
