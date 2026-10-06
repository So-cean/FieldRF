"""Restormer inference — uses self-contained checkpoint format."""

import torch

from .base_inference import BaseInference, extract_generator_state
from ..models.networks_restormer import RestormerGenerator, RestormerGeneratorNoSkip


class RestormerInference(BaseInference):
    """Inference for Restormer generator."""

    def load_generator(self):
        state = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False,
        )
        kw = state["generator_config"]["kwargs"]
        _Gen = RestormerGenerator if kw.get("use_skip", True) else RestormerGeneratorNoSkip
        model = _Gen(
            input_nc=kw["input_nc"],
            output_nc=kw["output_nc"],
            ngf=kw.get("ngf", 48),
            encoder_blocks=kw.get("encoder_blocks", [2, 2, 4, 6]),
            num_heads=kw.get("num_heads", [1, 2, 4, 8]),
            use_tanh=kw.get("use_tanh", True),
        )
        gen_state = extract_generator_state(state)
        model.load_state_dict(gen_state)
        print(f"Loaded Restormer generator from: {self.checkpoint_path}")
        return model


__all__ = [
    "RestormerInference",
]
