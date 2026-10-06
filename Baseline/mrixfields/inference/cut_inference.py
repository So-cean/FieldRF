"""CUT inference (ResnetGenerator) — uses self-contained checkpoint format."""

import torch

from .base_inference import BaseInference, extract_generator_state
from ..models.networks_cut import ResnetGenerator, get_norm_layer


class CUTInference(BaseInference):
    """Inference for CUT (ResnetGenerator)."""

    def load_generator(self):
        state = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False,
        )
        if "generator_config" in state and state["generator_config"].get("kwargs"):
            kw = state["generator_config"]["kwargs"].copy()
        else:
            kw = self.cfg.get("model", {}).copy()
            kw.setdefault("norm_layer", "instance")
            kw.setdefault("use_tanh", not self.cfg.get("data", {}).get("normalize", False))
        norm_layer = get_norm_layer(kw.pop("norm_layer", "instance"))
        model = ResnetGenerator(
            input_nc=kw["input_nc"],
            output_nc=kw["output_nc"],
            ngf=kw.get("ngf", 64),
            n_blocks=kw.get("n_blocks", 9),
            norm_layer=norm_layer,
            use_dropout=kw.get("use_dropout", False),
            padding_type=kw.get("padding_type", "reflect"),
            use_tanh=kw.get("use_tanh", True),
            skip_dropout=kw.get("skip_dropout", 0.0),
        )
        gen_state = extract_generator_state(state)
        model.load_state_dict(gen_state)
        print(f"Loaded CUT generator from: {self.checkpoint_path}")
        return model


__all__ = [
    "CUTInference",
]
