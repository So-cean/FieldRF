"""FieldRF network components: gradient-safe MAISI UNet and source adapter."""

import torch
import torch.nn as nn
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.utils.type_conversion import convert_to_tensor


class DiffusionModelUNetMaisiGradSafe(DiffusionModelUNetMaisi):
    """Out-of-place adds for down_block/mid_block residuals (autograd-safe).

    The upstream MAISI UNet uses in-place ``h += residual`` which breaks
    autograd when the residual has gradient (e.g. our ControlNet projections).
    This override replaces them with out-of-place ``h = h + residual``.
    """

    def _apply_down_blocks(self, h, emb, context, down_block_additional_residuals):
        down_block_res_samples = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)

        if down_block_additional_residuals is not None:
            if len(down_block_additional_residuals) != len(down_block_res_samples):
                raise ValueError(
                    f"down_block_additional_residuals has {len(down_block_additional_residuals)} "
                    f"elements but {len(down_block_res_samples)} expected."
                )
            # OUT-OF-PLACE add (was in-place upstream).
            down_block_res_samples = [
                res + add for res, add in zip(down_block_res_samples, down_block_additional_residuals)
            ]
        return h, down_block_res_samples

    def forward(self, x, timesteps, context=None, class_labels=None,
                spacing_tensor=None, down_block_additional_residuals=None,
                mid_block_additional_residual=None,
                source_additional_residual=None, **kwargs):
        h = self.conv_in(x)
        if source_additional_residual is not None:
            if source_additional_residual.shape != h.shape:
                raise ValueError(
                    "source_additional_residual shape "
                    f"{source_additional_residual.shape} != conv_in output {h.shape}"
                )
            h = h + source_additional_residual
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, context, None, spacing_tensor)
        h, _updated = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual
        h = self._apply_up_blocks(h, emb, context, _updated)
        h = self.out(h)
        return convert_to_tensor(h)


class SourceLatentAdapter(nn.Module):
    """Inject source anatomy without expanding the pretrained UNet input.

    Per-channel spatial normalization removes the easiest field/intensity
    shortcut.  The final convolution is zero initialized, so an adapter model
    loaded from identity RF is functionally identical to identity RF before Stage 2 updates.
    """

    def __init__(self, in_channels=4, out_channels=64, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_src):
        dims = tuple(range(2, z_src.ndim))
        mean = z_src.mean(dim=dims, keepdim=True)
        var = z_src.var(dim=dims, keepdim=True, unbiased=False)
        z_src = (z_src - mean) * torch.rsqrt(var + 1e-6)
        return self.net(z_src)
