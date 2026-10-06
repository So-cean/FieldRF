"""CycleGAN inference (ResnetGenerator A -> B)."""

import torch

from .base_inference import BaseInference, extract_generator_state
from ..models.networks_cyclegan import ResnetGenerator


class CycleGANInference(BaseInference):
    """Inference for CycleGAN (G_AB: domain A -> domain B)."""

    def load_generator(self):
        model_cfg = self.cfg["model"]

        model = ResnetGenerator(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ngf=model_cfg.get("ngf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
        )

        state_dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        gen_state = extract_generator_state(state_dict)
        model.load_state_dict(gen_state)
        print(f"Loaded CycleGAN generator (A->B) from: {self.checkpoint_path}")
        return model

__all__ = [
    "CycleGANInference",
]
