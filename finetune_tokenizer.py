"""
Adapted from NeMo's speech_to_text_finetune.py for AI4Bharat IndicConformer on H100.
"""
import os
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from huggingface_hub import hf_hub_download

# --- USER CONFIGURATION ---
# Model Details
HF_REPO_ID = "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large"
HF_FILENAME = "indicconformer_stt_ta_hybrid_rnnt_large.nemo"

# Data Paths (Ensure these are the CLEAN versions)
TRAIN_MANIFEST = "train_manifest_clean.json"
TEST_MANIFEST = "test_manifest_clean.json"

# H100 Hyperparameters
BATCH_SIZE = 64        # Increase to 128 if memory allows
NUM_WORKERS = 16       # 16-32 for H100
PRECISION = "bf16-mixed" # Native H100 precision
MAX_EPOCHS = 100
LEARNING_RATE = 1e-5   # Low LR for dialect adaptation

def get_base_model(trainer: pl.Trainer, cfg: DictConfig) -> ASRModel:
    """
    Downloads and restores the AI4Bharat model.
    """
    # 1. Download if not exists
    logging.info(f"Checking for model file {HF_FILENAME}...")
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        logging.info(f"Model found at: {model_path}")
    except Exception as e:
        raise ValueError(f"Could not download model from HF: {e}")

    # 2. Restore
    logging.info(f"Restoring model from {model_path}...")
    # Explicitly using the Hybrid class ensures correct loading for this specific model
    asr_model = EncDecHybridRNNTCTCModel.restore_from(restore_path=model_path)
    
    asr_model.set_trainer(trainer)
    return asr_model

def setup_dataloaders(asr_model: ASRModel, cfg: DictConfig) -> ASRModel:
    """
    Sets up the training and validation dataloaders.
    """
    # Create Data Configs
    data_cfg = {
        'manifest_filepath': TRAIN_MANIFEST,
        'sample_rate': 16000,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'shuffle': True,
        'pin_memory': True,
        'use_start_end_token': False,
        'trim_silence': True,
        'max_duration': 20.0,
        'min_duration': 0.5
    }
    
    val_cfg = data_cfg.copy()
    val_cfg['manifest_filepath'] = TEST_MANIFEST
    val_cfg['shuffle'] = False

    # Apply to model
    # We use OmegaConf to create specific config objects expected by NeMo
    with open_dict(asr_model.cfg):
        asr_model.cfg.train_ds = OmegaConf.create(data_cfg)
        asr_model.cfg.validation_ds = OmegaConf.create(val_cfg)
        
        # Dialect Adaptation: Lower LR
        asr_model.cfg.optim.lr = LEARNING_RATE
        asr_model.cfg.optim.sched.min_lr = 1e-6
        
        # Dialect Adaptation: Stronger SpecAugment
        if 'spec_augment' in asr_model.cfg:
            asr_model.cfg.spec_augment.freq_masks = 2
            asr_model.cfg.spec_augment.time_masks = 5
            asr_model.cfg.spec_augment.freq_width = 27

    # Setup within the model
    asr_model.setup_training_data(train_data_config=asr_model.cfg.train_ds)
    asr_model.setup_validation_data(val_data_config=asr_model.cfg.validation_ds)
    asr_model.setup_optimization(optim_config=asr_model.cfg.optim)

    return asr_model

def main():
    # 1. Create a configuration similar to what Hydra would provide
    cfg = OmegaConf.create({
        "trainer": {
            "devices": 1,
            "accelerator": "gpu",
            "max_epochs": MAX_EPOCHS,
            "precision": PRECISION,
            "accumulate_grad_batches": 1,
            "enable_checkpointing": False, # Handled by exp_manager
            "logger": False,               # Handled by exp_manager
            "log_every_n_steps": 5,
            "val_check_interval": 0.5,
            "gradient_clip_val": 1.0,
        },
        "exp_manager": {
            "name": "Tamil_H100_Finetune",
            "save_best_model": True,
            "explicit_log_dir": "nemo_experiments/Tamil_H100_Run",
            "checkpoint_callback_params": {
                "monitor": "val_wer",
                "mode": "min",
                "save_top_k": 3,
                "always_save_nemo": True
            }
        }
    })

    logging.info(f'Configuration Setup Complete for H100')

    # 2. Initialize Trainer
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    # 3. Get Model
    asr_model = get_base_model(trainer, cfg)

    # 4. Setup Data & Optimizers
    asr_model = setup_dataloaders(asr_model, cfg)

    # 5. Start Training
    logging.info("Starting Auto-Finetuning...")
    trainer.fit(asr_model)
    
    logging.info("Training Finished.")

if __name__ == '__main__':
    main()
