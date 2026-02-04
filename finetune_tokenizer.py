import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

# 1. Configuration
MODEL_NAME = "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large"
TRAIN_MANIFEST = "train_manifest_clean.json"
TEST_MANIFEST = "test_manifest_clean.json"
# H100 Specifics
BATCH_SIZE = 64  # Start here. If H100 memory allows, go to 128.
PRECISION = "bf16-mixed" # H100 native precision

def main():
    # 2. Load Model
    print(f"Loading {MODEL_NAME}...")
    asr_model = EncDecHybridRNNTCTCModel.from_pretrained(model_name=MODEL_NAME)

    # 3. Setup Data Loaders
    # We create a config dictionary
    cfg = OmegaConf.create({
        'manifest_filepath': TRAIN_MANIFEST,
        'sample_rate': 16000,
        'batch_size': BATCH_SIZE, 
        'shuffle': True, 
        'num_workers': 16, # High worker count for H100 data feeding
        'pin_memory': True,
        'use_start_end_token': False,
        'trim_silence': True,
        'max_duration': 20.0, 
        'min_duration': 0.5 
    })

    # Setup Training Data
    asr_model.setup_training_data(train_data_config=cfg)

    # Setup Validation Data
    test_cfg = cfg.copy()
    test_cfg['manifest_filepath'] = TEST_MANIFEST
    test_cfg['shuffle'] = False
    asr_model.setup_validation_data(val_data_config=test_cfg)

    # 4. Dialect Adaptation Configuration
    # We must modify the internal config to increase augmentation
    # Using open_dict to allow modification of the struct
    with open_dict(asr_model.cfg):
        # OPTIMIZER: Lower LR because we want to adapt, not rewrite the brain
        asr_model.cfg.optim.lr = 1e-5 
        asr_model.cfg.optim.sched.min_lr = 1e-6
        
        # SPEC AUGMENT: Increase masking to force learning dialect features
        if 'spec_augment' in asr_model.cfg:
            asr_model.cfg.spec_augment.freq_masks = 2
            asr_model.cfg.spec_augment.time_masks = 5
            asr_model.cfg.spec_augment.freq_width = 27
            asr_model.cfg.spec_augment.time_width = 0.05

    # Apply optimization settings
    asr_model.setup_optimization(optim_config=asr_model.cfg.optim)

    # 5. Trainer Setup for H100
    trainer = pl.Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=100, # More epochs because learning rate is lower
        precision=PRECISION, 
        accumulate_grad_batches=1,
        enable_checkpointing=True,
        logger=False,
        log_every_n_steps=5,
        val_check_interval=0.5, # Validate twice per epoch to catch overfitting specific to dialect
        gradient_clip_val=1.0
    )

    # 6. Run Training
    print("Starting Dialect Adaptation...")
    trainer.fit(asr_model)

    # 7. Save
    save_path = "indicconformer_tamil_dialect_finetuned.nemo"
    asr_model.save_to(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()