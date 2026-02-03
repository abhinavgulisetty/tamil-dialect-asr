import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.losses.rnnt import RNNTLoss


if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. This script expects an H100 GPU.")

print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
torch.set_float32_matmul_precision('medium') 

class PrintEpochCallback(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"\nEpoch {trainer.current_epoch + 1} finished.")

model_name = "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large"
print(f"Loading model: {model_name}")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
print("Model loaded successfully.")

train_manifest = os.path.abspath("train_manifest_mono.json")
val_manifest = os.path.abspath("val_manifest_mono.json")

if not os.path.exists(train_manifest):
    raise FileNotFoundError(f"Train manifest not found at {train_manifest}")
if not os.path.exists(val_manifest):
    raise FileNotFoundError(f"Val manifest not found at {val_manifest}")

config = asr_model.cfg

BATCH_SIZE = 32  
NUM_WORKERS = 16 

config.train_ds.manifest_filepath = train_manifest
config.train_ds.batch_size = BATCH_SIZE
config.train_ds.num_workers = NUM_WORKERS
config.train_ds.pin_memory = True
config.train_ds.max_duration = 16.0 
config.train_ds.is_concat = False 
config.train_ds.shuffle = True

config.validation_ds.manifest_filepath = val_manifest
config.validation_ds.batch_size = BATCH_SIZE
config.validation_ds.num_workers = NUM_WORKERS
config.validation_ds.pin_memory = True

config.optim.lr = 0.001 
config.optim.betas = [0.9, 0.98] 
config.optim.weight_decay = 1e-3
config.optim.sched = {
    "name": "CosineAnnealing",
    "warmup_steps": 1000, 
    "min_lr": 1e-6
}

config.loss.loss_name = "pytorch"

asr_model.cfg = config

if hasattr(asr_model, 'loss'):
    new_loss = RNNTLoss(
        num_classes=asr_model.loss._blank,
        reduction=asr_model.loss.reduction,
        loss_name="pytorch",
        loss_kwargs={}
    )
    asr_model.loss = new_loss
    print(f"Loss updated to: {type(asr_model.loss._loss)}")

asr_model.setup_training_data(train_data_config=config.train_ds)
asr_model.setup_validation_data(val_data_config=config.validation_ds)

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=50, 
    precision="bf16-mixed", 
    accumulate_grad_batches=1,
    enable_checkpointing=True,
    callbacks=[PrintEpochCallback()],
    logger=False, 
    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    default_root_dir="nemo_experiments/indic_finetune_h100"
)

asr_model.set_trainer(trainer)

if config.spec_augment is not None and not hasattr(asr_model, 'spec_augmentation'):
     from nemo.collections.asr.modules import SpectrogramAugmentation
     asr_model.spec_augmentation = SpectrogramAugmentation(**config.spec_augment)
     print("SpecAugment initialized.")

print("Starting training...")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Precision: {trainer.precision}")
print(f"SpecAugment: {'Enabled' if hasattr(asr_model, 'spec_augmentation') else 'Disabled'}")

trainer.fit(asr_model)

print("Training finished. Saving model...")
os.makedirs("fine_tuned_asr_model", exist_ok=True)
save_path = "fine_tuned_asr_model/indic_conformer_tamil_fist.nemo"
asr_model.save_to(save_path)
print(f"Model saved to {save_path}")
