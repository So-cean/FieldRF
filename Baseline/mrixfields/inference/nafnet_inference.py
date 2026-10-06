"""NAFNet inference — uses self-contained checkpoint format."""

import torch

from .base_inference import BaseInference, extract_generator_state
from ..models.networks_nafnet import NAFNetGenerator


class NAFNetInference(BaseInference):
    """Inference for NAFNet generator."""

    def load_generator(self):
        state = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False,
        )
        kw = state["generator_config"]["kwargs"]
        model = NAFNetGenerator(
            input_nc=kw["input_nc"],
            output_nc=kw["output_nc"],
            ngf=kw.get("ngf", 32),
            enc_blk_nums=kw.get("enc_blk_nums", [2, 2, 4, 8]),
            middle_blk_num=kw.get("middle_blk_num", 12),
            dec_blk_nums=kw.get("dec_blk_nums", [2, 2, 2, 2]),
            DW_Expand=kw.get("DW_Expand", 2),
            FFN_Expand=kw.get("FFN_Expand", 2),
            drop_out_rate=kw.get("drop_out_rate", 0.),
            use_tanh=kw.get("use_tanh", True),
        )
        gen_state = extract_generator_state(state)
        model.load_state_dict(gen_state)
        print(f"Loaded NAFNet generator from: {self.checkpoint_path}")
        return model


__all__ = [
    "NAFNetInference",
]
