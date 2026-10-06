"""UNSB inference (multi-step Schrodinger Bridge refinement)."""

import torch
import numpy as np

from .base_inference import BaseInference, extract_generator_state
from ..models.networks_unsb import ResnetGenerator_ncsn


class UNSBInference(BaseInference):
    """Inference for UNSB with multi-step refinement.

    UNSB generates images via successive refinement (NFE steps).
    The final output quality improves with more steps, but may
    over-translate at very high NFE.
    """

    def __init__(self, cfg, checkpoint_path, device, num_steps=None, source_field=None):
        self.num_steps = num_steps
        super().__init__(cfg, checkpoint_path, device, source_field=source_field)

    def load_generator(self):
        model_cfg = self.cfg["model"]

        model = ResnetGenerator_ncsn(
            input_nc=model_cfg.get("input_nc", 1),
            output_nc=model_cfg.get("output_nc", 1),
            ngf=model_cfg.get("ngf", 64),
            n_blocks=model_cfg.get("n_blocks", 9),
            n_mlp=model_cfg.get("n_mlp", 3),
            z_dim=model_cfg.get("z_dim", 256),
            t_emb_dim=model_cfg.get("t_emb_dim", 256),
        )

        state_dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        gen_state = extract_generator_state(state_dict)
        model.load_state_dict(gen_state)
        print(f"Loaded UNSB generator from: {self.checkpoint_path}")
        return model

    def predict(self, volume, slice_axis=2, crop_size=None, batch_size=1):
        """Run inference with multi-step Schrodinger Bridge refinement.

        Returns the final refined output (last NFE step).
        """
        from .base_inference import predict_volume_slice_by_slice

        model_cfg = self.cfg["model"]
        T = self.num_steps or model_cfg.get("num_timesteps", 5)
        tau = model_cfg.get("tau", 0.01)

        # Reconstruct time schedule
        incs = np.array([0] + [1.0 / (i + 1) for i in range(T - 1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times, dtype=torch.float32, device=self.device)

        def _infer_slice(tensor):
            """tensor: (B, 1, H, W) on device."""
            B = tensor.size(0)
            Xt = tensor
            for t in range(T):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).view(1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).view(1, 1, 1, 1)
                    noise = torch.randn_like(Xt)
                    Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * noise

                t_idx = (t * torch.ones(B, device=tensor.device)).long()
                z = torch.randn(B, model_cfg.get("z_dim", 256), device=tensor.device)
                Xt_1 = self.model(Xt, t_idx, z)
            return Xt_1

        return predict_volume_slice_by_slice(
            _UNSBWrap(self.model, T, tau, times),
            volume, self.device,
            slice_axis=slice_axis, crop_size=crop_size, batch_size=batch_size,
        )


class _UNSBWrap(torch.nn.Module):
    """Wrapper to make UNSB inference compatible with predict_volume_slice_by_slice."""

    def __init__(self, model, T, tau, times):
        super().__init__()
        self.model = model
        self.T = T
        self.tau = tau
        self.register_buffer("times", times)

    def forward(self, x):
        B = x.size(0)
        device = x.device
        Xt = x
        for t in range(self.T):
            if t > 0:
                delta = self.times[t] - self.times[t - 1]
                denom = self.times[-1] - self.times[t - 1]
                inter = (delta / denom).view(1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).view(1, 1, 1, 1)
                noise = torch.randn_like(Xt)
                Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * self.tau).sqrt() * noise

            t_idx = (t * torch.ones(B, device=device)).long()
            z = torch.randn(B, self.model.z_dim, device=device)
            Xt_1 = self.model(Xt, t_idx, z)
        return Xt_1

__all__ = [
    "UNSBInference",
]
