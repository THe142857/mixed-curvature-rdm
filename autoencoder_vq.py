from autoencoder_perceptive import VQModel
from omegaconf import OmegaConf
from main import instantiate_from_config
import torch

checkpoint = torch.load("model.ckpt", map_location=torch.device('cpu'))
config = OmegaConf.load("vq-f8/config.yaml")
model_vq = instantiate_from_config(config.model)
model_vq.load_state_dict(checkpoint["state_dict"], strict=False)