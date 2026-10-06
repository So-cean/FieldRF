"""ITTR inference (ITTRGenerator with DPSA blocks)."""

import torch

from .base_inference import BaseInference, extract_generator_state
from ..models.networks_ittr import ITTRGenerator


class ITTRInference(BaseInference):
    """Inference for ITTR (Image-to-Image Translation with Transformers)."""

    def load_generator(self):
        model_cfg = self.cfg["model"]

        # Infer generator config (checkpoint may have generator_config)
        gen_cfg = self.cfg.get("generator_config", {})
        gen_kwargs = gen_cfg.get("kwargs", {})

        model = ITTRGenerator(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ngf=model_cfg.get("ngf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
            dim_head=model_cfg.get("dim_head", 32),
            heads=model_cfg.get("heads", 8),
            attn_height_top_k=model_cfg.get("attn_height_top_k", 16),
            attn_width_top_k=model_cfg.get("attn_width_top_k", 16),
            attn_dropout=model_cfg.get("attn_dropout", 0.0),
            ff_dropout=model_cfg.get("ff_dropout", 0.0),
            ff_mult=model_cfg.get("ff_mult", 4),
            use_tanh=gen_kwargs.get("use_tanh", model_cfg.get("use_tanh", True)),
            use_conv_transpose=gen_kwargs.get("use_conv_transpose", model_cfg.get("use_conv_transpose", False)),
        )

        state_dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        gen_state = extract_generator_state(state_dict)
        model.load_state_dict(gen_state)
        print(f"Loaded ITTR generator from: {self.checkpoint_path}")
        return model

__all__ = [
    "ITTRInference",
]
