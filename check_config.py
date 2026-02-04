from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

# Load the local file
model = EncDecHybridRNNTCTCModel.restore_from("indicconformer_stt_ta_hybrid_rnnt_large.nemo")

# Save the internal config to a readable YAML file
model.save_to("my_debug_model.nemo") # This repacks it, but to see config:
from omegaconf import OmegaConf
OmegaConf.save(model.cfg, "extracted_config.yaml")

print("Config saved to extracted_config.yaml")
