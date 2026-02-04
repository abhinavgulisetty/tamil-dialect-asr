import os
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from huggingface_hub import hf_hub_download

# --- USER CONFIGURATION ---
REPO_ID = "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large"
FILENAME = "indicconformer_stt_ta_hybrid_rnnt_large.nemo" 

TRAIN_MANIFEST = "train_manifest_clean.json"
TEST_MANIFEST = "test_manifest_clean.json"

# H100 Settings
BATCH_SIZE = 64
ACCUMULATE_GRAD_BATCHES = 1
MAX_EPOCHS = 100
PRECISION = "bf16-mixed" # Use '16-mixed' if bf16 throws errors, but H100 supports bf16
NUM_WORKERS = 16

def main():
    # 1. Download Model
    logging.info(f"Checking for model file {FILENAME}...")
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        logging.info(f"Model found at: {model_path}")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        return

    # 2. Load the Pre-trained Model
    logging.info("Restoring model from checkpoint...")
    asr_model = EncDecHybridRNNTCTCModel.restore_from(restore_path=model_path)

    # 3. Configure Data Loaders
    logging.info(f"Setting up data loaders with Batch Size: {BATCH_SIZE}")
    
    # Common Data Config
    data_config = {
        'sample_rate': 16000,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'pin_memory': True,
        'use_start_end_token': False,
        'trim_silence': True,
        'max_duration': 20.0,
        'min_duration': 0.5
    }

    # Train Config
    train_cfg = data_config.copy()
    train_cfg['manifest_filepath'] = TRAIN_MANIFEST
    train_cfg['shuffle'] = True
    train_cfg['is_tarred'] = False # Ensure this is false for JSON manifests

    # Validation Config
    val_cfg = data_config.copy()
    val_cfg['manifest_filepath'] = TEST_MANIFEST
    val_cfg['shuffle'] = False

    # Apply Configs to Model
    # We use open_dict to safely modify the internal config structure
    with open_dict(asr_model.cfg):
        # Update Data Configs
        asr_model.cfg.train_ds = OmegaConf.create(train_cfg)
        asr_model.cfg.validation_ds = OmegaConf.create(val_cfg)
        
        # Update Optimizer (Dialect Adaptation Strategy: Low LR)
        asr_model.cfg.optim.lr = 1e-5
        asr_model.cfg.optim.sched.min_lr = 1e-6
        asr_model.cfg.optim.weight_decay = 0.001

        # Update SpecAugment (Robustness Strategy)
        if 'spec_augment' in asr_model.cfg:
            asr_model.cfg.spec_augment.freq_masks = 2
            asr_model.cfg.spec_augment.time_masks = 5
            asr_model.cfg.spec_augment.freq_width = 27
            asr_model.cfg.spec_augment.time_width = 0.05
    
    # Setup Data & Optimization inside NeMo model
    asr_model.setup_training_data(train_data_config=asr_model.cfg.train_ds)
    asr_model.setup_validation_data(val_data_config=asr_model.cfg.validation_ds)
    asr_model.setup_optimization(optim_config=asr_model.cfg.optim)

    # 4. Initialize Trainer
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=MAX_EPOCHS,
        precision=PRECISION,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        val_check_interval=0.5,
        enable_checkpointing=False, # We use exp_manager instead
        logger=False # We use exp_manager
    )

    # 5. Setup Experiment Manager (This handles checkpointing, logging, etc.)
    exp_config = OmegaConf.create({
        "name": "Tamil_Dialect_Finetune",
        "save_best_model": True,
        "explicit_log_dir": "nemo_experiments/Tamil_H100_Run",
        "checkpoint_callback_params": {
            "monitor": "val_wer",
            "mode": "min",
            "save_top_k": 3,
            "always_save_nemo": True # Important: Saves final .nemo file
        }
    })
    exp_manager(trainer, exp_config)

    # 6. Start Training
    logging.info("Starting Training...")
    trainer.fit(asr_model)

    logging.info("Training Finished.")
    # The final model will be saved by exp_manager in nemo_experiments/Tamil_H100_Run/checkpoints/

if __name__ == '__main__':
    main()
