"""FieldRF network assembly and velocity prediction.

Training paths, numerical integration and checkpoint I/O live in dedicated
modules. Thin model methods preserve existing training and inference entry
points, while registered submodules retain their checkpoint parameter names.
"""

import torch
import torch.nn as nn
from monai.networks.nets import ControlNet

from .networks import DiffusionModelUNetMaisiGradSafe, SourceLatentAdapter
from . import checkpoint, rectified_flow, sampling
# Retain existing helper imports for external scripts; maintained callers use
# the defining modules directly.
from .rectified_flow import timestep_transform
from utils.domains import FIELDS, MODALITIES, NUM_CLASSES, get_class_label
from utils.diagnostics import controlnet_norms, source_channel_norms


class FieldRFModel(nn.Module):
    """Shared anatomy-conditioned rectified flow across MRI domains.

    Owns the MAISI UNet, Brain-ID ControlNet and optional source-latent adapter.
    ``source_injection`` selects no source path, channel concatenation or a
    zero-initialized residual adapter. ``forward`` predicts data-minus-endpoint
    velocity. Training paths, inversion, sampling and loading methods delegate
    to the modules that implement those responsibilities.
    """

    def __init__(
        self,
        # UNet
        unet_channels=(64, 128, 256, 512),
        num_res_blocks=(2, 2, 2, 2),
        attention_levels=(False, False, True, True),
        num_head_channels=(0, 0, 32, 32),
        norm_num_groups=32,
        norm_eps=1e-6,
        num_class_embeds=NUM_CLASSES,
        # Brain-ID feat4
        feat4_channels=64,
        conditioning_embedding_num_channels=(64, 64),
        # Latent
        z_channels=4,
        # RF
        num_train_timesteps=1000,
        scale_factor=1.0,
        use_source_latent=False,
        source_injection=None,
        source_adapter_channels=16,
        use_flash_attention=True,
        resblock_updown=False,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.feat4_channels = feat4_channels
        self.scale_factor = scale_factor
        self.num_train_timesteps = num_train_timesteps
        # MAISI RF weights use residual resampling; legacy FieldRF weights
        # used convolutional resampling. Checkpoint loaders infer this flag.
        self.resblock_updown = bool(resblock_updown)
        self.conditioning_embedding_num_channels = tuple(
            int(value) for value in conditioning_embedding_num_channels
        )
        if not self.conditioning_embedding_num_channels:
            raise ValueError("conditioning_embedding_num_channels cannot be empty")
        self.conditioning_downsample_factor = 2 ** (
            len(self.conditioning_embedding_num_channels) - 1
        )
        if source_injection is None:
            source_injection = "concat" if use_source_latent else "none"
        if source_injection not in {"none", "concat", "adapter"}:
            raise ValueError(
                "source_injection must be one of none/concat/adapter, got "
                f"{source_injection!r}"
            )
        self.source_injection = source_injection
        self.use_source_latent = source_injection != "none"
        self.in_channels = z_channels * (2 if source_injection == "concat" else 1)

        # ── UNet (identity RF: z_t; source-conditioned RF: [z_t | z_src]) ── #
        self.unet = DiffusionModelUNetMaisiGradSafe(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=z_channels,
            num_channels=list(unet_channels),
            num_res_blocks=list(num_res_blocks),
            attention_levels=list(attention_levels),
            num_head_channels=list(num_head_channels),
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_class_embeds=num_class_embeds,
            include_spacing_input=True,
            use_flash_attention=use_flash_attention,
            resblock_updown=self.resblock_updown,
        )
        self.source_adapter = None
        if self.source_injection == "adapter":
            self.source_adapter = SourceLatentAdapter(
                in_channels=z_channels,
                out_channels=unet_channels[0],
                hidden_channels=source_adapter_channels,
            )

        # ── ControlNet (always 4ch z_t + feat4 conditioning) ── #
        # z_src is deliberately restricted to the main UNet. This keeps the
        # semantic feat4 path independent and makes its contribution measurable.
        # Legacy: feat4 192³ + [64,64] -> 96³. BrainID-HR native:
        # F96 96³ + [64] -> 96³. Neither route has a channel bottleneck.
        self.controlnet = ControlNet(
            spatial_dims=3,
            in_channels=z_channels,
            num_res_blocks=list(num_res_blocks),
            channels=list(unet_channels),
            attention_levels=list(attention_levels),
            norm_num_groups=norm_num_groups,
            num_head_channels=list(num_head_channels),
            use_flash_attention=use_flash_attention,
            num_class_embeds=num_class_embeds,
            conditioning_embedding_in_channels=feat4_channels,  # 64
            conditioning_embedding_num_channels=list(
                self.conditioning_embedding_num_channels
            ),
        )

        # Fixed spacing tensor (field info in class label, not spacing)
        self.register_buffer(
            "spacing_tensor", torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        )

    def _unet_input(self, z_t, z_src=None):
        """Build the identity RF/source-conditioned RF main-UNet input.

        z_src is in the same scaled latent space as z_t. Identity RF
        supervision passes None, which is exactly an all-zero source channel.
        """
        if self.source_injection != "concat":
            return z_t
        if z_src is None:
            z_src = torch.zeros_like(z_t)
        if z_src.shape != z_t.shape:
            raise ValueError(f"z_src shape {z_src.shape} != z_t shape {z_t.shape}")
        return torch.cat([z_t, z_src], dim=1)

    def _source_residual(self, z_t, t, z_src):
        """Return time-gated adapter residual for the conv-in feature map."""
        if self.source_injection != "adapter" or z_src is None:
            return None
        if z_src.shape != z_t.shape:
            raise ValueError(f"z_src shape {z_src.shape} != z_t shape {z_t.shape}")
        # Source anatomy is most useful while the RF state is noisy. Fading the
        # direct path near t=0 reduces source-field appearance leakage.
        gate = (t / self.num_train_timesteps).clamp(0, 1)
        gate = gate.view(-1, 1, 1, 1, 1).to(dtype=z_t.dtype)
        return self.source_adapter(z_src.to(dtype=z_t.dtype)) * gate

    def forward(self, z_t, t, class_label, feats, z_src=None):
        """Predict velocity v from (z_t, t, feats).

        Args:
            z_t: (B, 4, 96³) noisy latent (scaled by scale_factor).
            t: (B,) float timesteps [0, num_train_timesteps).
            class_label: (B,) target class (0-14).
            feats: list of 5 Brain-ID features [feat0..feat4].
            z_src: optional scaled source latent for concat/adapter injection.

        Returns:
            v_pred: (B, 4, 96³) predicted velocity.
        """
        feat4 = feats[-1].float()
        B = z_t.shape[0]
        expected_latent_shape = tuple(
            (length + self.conditioning_downsample_factor - 1)
            // self.conditioning_downsample_factor
            for length in feat4.shape[-3:]
        )
        if expected_latent_shape != tuple(z_t.shape[-3:]):
            raise ValueError(
                "Control feature/latent spatial contract mismatch: feature "
                f"{tuple(feat4.shape[-3:])} with conditioning pyramid "
                f"{self.conditioning_embedding_num_channels} produces "
                f"{expected_latent_shape}, but latent is {tuple(z_t.shape[-3:])}"
            )

        # ControlNet: z_t as main input, feat4 as conditioning
        down_res, mid_res = self.controlnet(
            z_t, timesteps=t,
            controlnet_cond=feat4, class_labels=class_label,
        )

        # UNet forward: predict velocity
        return self.unet(
            self._unet_input(z_t, z_src), timesteps=t,
            class_labels=class_label,
            spacing_tensor=self.spacing_tensor.expand(B, -1),
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            source_additional_residual=self._source_residual(z_t, t, z_src),
        )

    def sample_train_timesteps(
        self,
        z_0,
        endpoint_focus_probability=0.0,
        endpoint_min_fraction=0.7,
    ):
        """Sample MAISI-transformed training timesteps."""
        return rectified_flow.sample_train_timesteps(
            self,
            z_0,
            endpoint_focus_probability=endpoint_focus_probability,
            endpoint_min_fraction=endpoint_min_fraction,
        )

    def build_rf_path(self, z_0, endpoint, t):
        """Construct the RF state and data-minus-endpoint velocity."""
        return rectified_flow.build_rf_path(self, z_0, endpoint, t)

    def forward_rf(
        self,
        z_0,
        feats,
        class_label,
        t=None,
        z_src=None,
        endpoint=None,
        endpoint_focus_probability=0.0,
        endpoint_min_fraction=0.7,
    ):
        """Return predicted and target RF velocities; losses are external."""
        return rectified_flow.forward_rf(
            self,
            z_0,
            feats,
            class_label,
            t=t,
            z_src=z_src,
            endpoint=endpoint,
            endpoint_focus_probability=endpoint_focus_probability,
            endpoint_min_fraction=endpoint_min_fraction,
        )

    def invert_to_endpoint(
        self,
        z_0,
        feats,
        class_label,
        n_steps=4,
        solver="euler",
        velocity_model=None,
        gradient_steps=None,
    ):
        """Invert a scaled latent with Euler/Heun and optional gradient truncation."""
        return sampling.invert_to_endpoint(
            self,
            z_0,
            feats,
            class_label,
            n_steps=n_steps,
            solver=solver,
            velocity_model=velocity_model,
            gradient_steps=gradient_steps,
        )

    def sample_x0(self, feats, class_label, n_infer_steps=20, device=None, z_src=None):
        """Sample from noise and return an unscaled latent under no_grad."""
        return sampling.sample_x0(
            self,
            feats,
            class_label,
            n_infer_steps=n_infer_steps,
            device=device,
            z_src=z_src,
        )

    def sample_x0_from_endpoint(
        self,
        endpoint,
        feats,
        class_label,
        n_infer_steps=20,
    ):
        """Reverse a supplied scaled endpoint under no_grad; return an unscaled latent."""
        return sampling.sample_x0_from_endpoint(
            self,
            endpoint,
            feats,
            class_label,
            n_infer_steps=n_infer_steps,
        )

    def generate_train(self, feats, class_label, z_src, source_dropout_p=0.1,
                       t=None, n_infer_steps=1, return_diagnostics=False):
        """Generate a differentiable source-conditioned graft in scaled latent space."""
        return sampling.generate_train(
            self,
            feats,
            class_label,
            z_src,
            source_dropout_p=source_dropout_p,
            t=t,
            n_infer_steps=n_infer_steps,
            return_diagnostics=return_diagnostics,
        )

    @staticmethod
    def from_pretrained(
        ckpt_path,
        unet_channels=(64, 128, 256, 512),
        num_res_blocks=(2, 2, 2, 2),
        attention_levels=(False, False, True, True),
        num_head_channels=(0, 0, 32, 32),
        norm_num_groups=32,
        norm_eps=1e-6,
        num_class_embeds=NUM_CLASSES,
        feat4_channels=64,
        conditioning_embedding_num_channels=(64, 64),
        z_channels=4,
        num_train_timesteps=1000,
        scale_factor=1.0,
        use_source_latent=False,
        source_injection=None,
        source_adapter_channels=16,
        use_flash_attention=True,
        device="cpu",
        load_ema=False,
        **kwargs,
    ):
        """Load pretrained or full RF/EMA weights with architecture adaptation."""
        return checkpoint.load_pretrained(
            FieldRFModel,
            ckpt_path,
            unet_channels=unet_channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_class_embeds=num_class_embeds,
            feat4_channels=feat4_channels,
            conditioning_embedding_num_channels=conditioning_embedding_num_channels,
            z_channels=z_channels,
            num_train_timesteps=num_train_timesteps,
            scale_factor=scale_factor,
            use_source_latent=use_source_latent,
            source_injection=source_injection,
            source_adapter_channels=source_adapter_channels,
            use_flash_attention=use_flash_attention,
            device=device,
            load_ema=load_ema,
            **kwargs,
        )

    @staticmethod
    def _load_bridge_state(bridge, state_dict):
        """Validate RF weights, allowing only an absent source adapter."""
        return checkpoint.load_bridge_state(bridge, state_dict)

    @staticmethod
    def _adapt_source_channels(state_dict, bridge):
        """Zero-expand source channels when the requested architecture requires it."""
        return checkpoint.adapt_source_channels(state_dict, bridge)
