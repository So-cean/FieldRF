"""FieldRF-V7: Rectified Flow bridge + ControlNet (feat4), MAISI-aligned.

Architecture:
  - Rectified Flow (RF) forward: linear interpolation between z_0 and noise.
  - Velocity (v) prediction via UNet + ControlNet(feat4).
  - Euler ODE sampling (deterministic reverse), no ancestral noise.
  - Timestep-transform (scale=1.4, base=32³) aligned with MAISI recipe.
  - Conditioning_embedding [64,64] (was [16,64] in V6).

Key differences from V6:
  - SelfRDB schedule removed entirely (no bridge soft prior, no endpoint).
    - RF forward: z_t = (1-t/T)*z_0 + (t/T)*noise (MAISI convention).
    - v = z_0 - noise (NOT noise - z_0). VERIFIED against MAISI
      rectified_flow.py: model_gt = images - noise. Same sign convention.
    - t is FLOAT, passed directly to UNet.
  - ControlNet conditioning_embedding_num_channels = [64, 64].
  - feat4_downsample / endpoint_proj removed (no source projection).
  - mu_x0, mu_y, std schedule buffers removed.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.nets import ControlNet
from monai.utils.type_conversion import convert_to_tensor


# ── Timestep transform (MAISI rectified_flow.py, verbatim) ── #

def timestep_transform(t, input_img_size_numel, base_img_size_numel=32768,
                       scale=1.0, num_train_timesteps=1000, spatial_dim=3):
    """MAISI timestep remap: resolution-driven sigmoid-like correction.

    Args:
        t: (B,) float timesteps in [0, num_train_timesteps).
        input_img_size_numel: total voxels in the latent (e.g. 96*96*96=884736).
        base_img_size_numel: reference voxel count (32*32*32=32768).
        scale: scaling factor for the remap (MAISI default 1.4).
        num_train_timesteps: max timestep (1000).
        spatial_dim: 3 for 3D.
    Returns:
        new_t: (B,) transformed timesteps, same range as input.
    """
    t = t / num_train_timesteps  # normalize to [0, 1]
    ratio_space = (input_img_size_numel / base_img_size_numel) ** (1.0 / spatial_dim)
    ratio = ratio_space * scale
    # Remap: new_t = ratio * t / (1 + (ratio - 1) * t)
    new_t = ratio * t / (1 + (ratio - 1) * t)
    new_t = new_t * num_train_timesteps
    return new_t


# ── Class label mapping (same as V3/V6) ── #

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
NUM_CLASSES = len(MODALITIES) * len(FIELDS)  # 15


def get_class_label(modality, field):
    return MODALITIES.index(modality) * len(FIELDS) + FIELDS.index(field)


# ── Grad-safe UNet wrapper (for ControlNet residual injection) ── #

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
    loaded from V7a is functionally identical to V7a before Stage 2 updates.
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


# ── V7a Bridge: Rectified Flow + ControlNet ── #

class BridgeRFV7(nn.Module):
    """FieldRF-V7: Class-conditional latent diffusion with RF + ControlNet(feat4).

    Architecture:
      - V7a UNet: 4ch conv_in for z_t, pretrained init.
      - Legacy V7b concat: 8ch conv_in for [z_t | z_src], with zero-init
        source-channel weights when expanding a V7a checkpoint.
      - V7b adapter: the 4ch V7a conv_in remains intact and a zero-init source
        residual is added after conv_in.
      - ControlNet: monai ControlNet with a configurable conditioning pyramid.
        Legacy Brain-ID uses 192³ via [64,64] (one stride-2); BrainID-HR uses
        native 96³ via [64] (no spatial downsample).
      - Random init ControlNet (DO NOT copy UNet weights).
      - Timestep-transform applied per MAISI recipe (scale=1.4, base=32³).

    Training:
      z_t = (1 - t/T) * z_0 + (t/T) * noise   (MAISI convention)
      v_pred = UNet(z_t, t, class, ControlNet(feat4))
      loss = L1(v_pred, v_target) where v_target = z_0 - noise

    Inference:
      Euler ODE from noise at t=T to z_0 at t=0.
      Deterministic, N=20 steps (configurable).

    No SelfRDB residue. Stage 2 adversarial training is implemented by the
    trainer; this module supplies the V7b source-latent path and graft sampler.
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
    ):
        super().__init__()
        self.z_channels = z_channels
        self.feat4_channels = feat4_channels
        self.scale_factor = scale_factor
        self.num_train_timesteps = num_train_timesteps
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

        # ── UNet (V7a: z_t; V7b: [z_t | z_src]) ── #
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

    # ── Noise prediction forward ── #

    def _unet_input(self, z_t, z_src=None):
        """Build the V7a/V7b main-UNet input.

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

    # ── RF forward (for training loop) ── #

    def sample_train_timesteps(
        self,
        z_0,
        endpoint_focus_probability=0.0,
        endpoint_min_fraction=0.7,
    ):
        """Sample transformed RF timesteps used by ``forward_rf``.

        ``endpoint_focus_probability`` optionally replaces part of the
        uniform raw-timestep batch with samples close to the noise endpoint.
        The returned values have already passed through MAISI's timestep
        transform and can be shared by paired source/target branches.
        """
        if not 0.0 <= endpoint_focus_probability <= 1.0:
            raise ValueError(
                "endpoint_focus_probability must be in [0, 1], got "
                f"{endpoint_focus_probability}"
            )
        if not 0.0 <= endpoint_min_fraction < 1.0:
            raise ValueError(
                "endpoint_min_fraction must be in [0, 1), got "
                f"{endpoint_min_fraction}"
            )

        batch = z_0.shape[0]
        raw_t = torch.rand((batch,), device=z_0.device)
        if endpoint_focus_probability > 0:
            use_endpoint = (
                torch.rand((batch,), device=z_0.device)
                < endpoint_focus_probability
            )
            endpoint_t = endpoint_min_fraction + (
                1.0 - endpoint_min_fraction
            ) * torch.rand((batch,), device=z_0.device)
            raw_t = torch.where(use_endpoint, endpoint_t, raw_t)
        raw_t = raw_t * self.num_train_timesteps

        input_img_size_numel = z_0.shape[2] * z_0.shape[3] * z_0.shape[4]
        return timestep_transform(
            raw_t,
            input_img_size_numel,
            base_img_size_numel=32768,
            scale=1.4,
            num_train_timesteps=self.num_train_timesteps,
            spatial_dim=3,
        )

    def build_rf_path(self, z_0, endpoint, t):
        """Build a linear RF state and its data-minus-endpoint velocity.

        Args:
            z_0: Clean, scaled image latent.
            endpoint: Noise or another target-compatible RF endpoint.
            t: MAISI-transformed timestep returned by
                :meth:`sample_train_timesteps`.
        """
        if endpoint.shape != z_0.shape:
            raise ValueError(
                f"endpoint shape {endpoint.shape} != z_0 shape {z_0.shape}"
            )
        if t.ndim != 1 or t.shape[0] != z_0.shape[0]:
            raise ValueError(
                f"t must have shape ({z_0.shape[0]},), got {tuple(t.shape)}"
            )
        endpoint = endpoint.to(device=z_0.device, dtype=z_0.dtype)
        timepoints = 1.0 - t / self.num_train_timesteps
        tp = timepoints.view(-1, 1, 1, 1, 1)
        z_t = tp * z_0 + (1.0 - tp) * endpoint
        v_target = z_0 - endpoint
        return z_t, v_target

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
        """RF forward: noise z_0, predict velocity v.

        If t is None, samples uniformly with timestep_transform.

        Args:
            z_0: (B, 4, 96³) clean latent (scaled).
            feats: list of 5 Brain-ID features.
            class_label: (B,) target class.
            t: optional pre-sampled, already transformed timesteps.
            z_src: optional scaled source latent. Identity supervision leaves
                this unset so the new V7b channels remain zero.
            endpoint: optional externally supplied RF endpoint. Passing the
                same deterministic endpoint to paired source/target branches
                establishes an explicit sample-wise coupling.
        Returns:
            v_pred: predicted velocity.
            v_target: ground-truth velocity.
            loss: L1 loss between prediction and target.
        """
        B = z_0.shape[0]
        device = z_0.device

        if t is None:
            t = self.sample_train_timesteps(
                z_0,
                endpoint_focus_probability=endpoint_focus_probability,
                endpoint_min_fraction=endpoint_min_fraction,
            )

        if endpoint is None:
            endpoint = torch.randn_like(z_0)
        z_t, v_target = self.build_rf_path(z_0, endpoint, t)

        # Predict velocity
        v_pred = self.forward(z_t, t, class_label, feats, z_src=z_src)

        # L1 loss (MAISI uses L1 even for v-pred)
        loss = F.l1_loss(v_pred, v_target)
        return v_pred, v_target, loss

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
        """Integrate a scaled image latent toward its learned RF endpoint.

        This method intentionally remains differentiable. Callers that only
        need a detached source endpoint should invoke it under
        ``torch.no_grad()``. Short differentiable integrations can be used for
        an optional explicit endpoint-consistency loss.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if solver not in {"euler", "heun"}:
            raise ValueError(f"Unsupported solver: {solver}")
        if gradient_steps is None:
            gradient_steps = n_steps
        gradient_steps = int(gradient_steps)
        if not 0 <= gradient_steps <= n_steps:
            raise ValueError(
                "gradient_steps must lie in [0, n_steps], got "
                f"{gradient_steps} for n_steps={n_steps}"
            )
        predict_velocity = self.forward if velocity_model is None else velocity_model

        batch = z_0.shape[0]
        if isinstance(class_label, int):
            class_label = torch.full(
                (batch,),
                class_label,
                device=z_0.device,
                dtype=torch.long,
            )
        else:
            class_label = class_label.to(device=z_0.device, dtype=torch.long)

        raw_schedule = torch.linspace(
            0,
            self.num_train_timesteps - 1,
            n_steps + 1,
            device=z_0.device,
        )
        schedule = timestep_transform(
            raw_schedule,
            int(np.prod(z_0.shape[2:])),
            base_img_size_numel=32768,
            scale=1.4,
            num_train_timesteps=self.num_train_timesteps,
            spatial_dim=3,
        )

        state = z_0
        for index in range(n_steps):
            current_t = schedule[index].expand(batch)
            next_t = schedule[index + 1].expand(batch)
            delta = (
                schedule[index + 1] - schedule[index]
            ) / self.num_train_timesteps
            # Full deployment uses several ODE steps, but retaining every
            # 96^3 activation is prohibitively expensive.  ``gradient_steps``
            # implements truncated BPTT: the detached prefix follows the
            # actual online vector field and the final steps remain
            # differentiable.  The default preserves the historical fully
            # differentiable behavior.
            keep_gradient = index >= n_steps - gradient_steps
            context = nullcontext() if keep_gradient else torch.no_grad()
            with context:
                velocity = predict_velocity(
                    state,
                    current_t,
                    class_label,
                    feats,
                )
                prediction = state - velocity * delta
                if solver == "euler":
                    state = prediction
                else:
                    next_velocity = predict_velocity(
                        prediction,
                        next_t,
                        class_label,
                        feats,
                    )
                    state = state - 0.5 * (velocity + next_velocity) * delta
            if not keep_gradient:
                state = state.detach()
        return state

    # ── Inference: Euler ODE ── #

    @torch.no_grad()
    def sample_x0(self, feats, class_label, n_infer_steps=20, device=None, z_src=None):
        """Euler ODE sampling: deterministic reverse from noise to x0.

        Args:
            feats: list of 5 Brain-ID features.
            class_label: (B,) or int target class.
            n_infer_steps: number of Euler steps (default 20).
            z_src: optional UNSCALED source latent for V7b conditioning.
        Returns:
            x0_pred: (B, 4, 96³) predicted target latent (UNSCALED, for VAE decode).
        """
        if device is None:
            device = next(self.parameters()).device

        B = feats[0].shape[0]
        if isinstance(class_label, int):
            class_label = torch.full((B,), class_label, device=device, dtype=torch.long)
        else:
            class_label = class_label.to(device)

        feat4 = feats[-1].float().to(device)
        z_src_scaled = None
        if z_src is not None:
            z_src_scaled = z_src.to(device).float() * self.scale_factor

        # Build inference timesteps: uniform from T-1 to 0
        # (T-1, not T, because training samples t ∈ [0, T) — querying at t=T
        # would be out-of-distribution. The model has seen t up to ~T-1.)
        N = n_infer_steps
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, N + 1, device=device
        )  # [T-1, ..., 0]

        # Apply timestep_transform to inference timesteps too (matching training).
        # CRITICAL: use LATENT spatial numel (96³), NOT feat4's 192³ — the transform
        # is a noise-schedule correction for the generation resolution, and the
        # generation happens at the latent's 96³, not feat4's 192³. Using 192³ here
        # would give ratio=8.4 vs training's 4.2, completely mismatching the
        # t-distribution the model was trained on.
        latent_shape = tuple(
            (length + self.conditioning_downsample_factor - 1)
            // self.conditioning_downsample_factor
            for length in feat4.shape[-3:]
        )
        input_img_size_numel = int(np.prod(latent_shape))
        timesteps = timestep_transform(
            timesteps, input_img_size_numel,
            base_img_size_numel=32768,
            scale=1.4,
            num_train_timesteps=self.num_train_timesteps,
            spatial_dim=3,
        )

        # Initial noise: sample from standard normal (t=T → pure noise)
        # At t=T: z_T ≈ noise (since timepoints ≈ 0)
        z_t = torch.randn(B, self.z_channels, *latent_shape, device=device)

        # Euler ODE: z_{t-dt} = z_t + v * dt
        for i in range(N):
            t_val = timesteps[i]  # scalar on device
            t_batch = t_val.expand(B)

            # Predict velocity
            v_pred = self.forward(
                z_t, t_batch, class_label, feats, z_src=z_src_scaled
            )

            # dt: difference to next timestep (in normalized units)
            dt = (timesteps[i] - timesteps[i + 1]) / self.num_train_timesteps

            # Euler step: z_{t-dt} = z_t + v * dt
            z_t = z_t + v_pred * dt

        # Final z_t ≈ z_0_pred (unscaled for VAE decode)
        return z_t / self.scale_factor

    @torch.no_grad()
    def sample_x0_from_endpoint(
        self,
        endpoint,
        feats,
        class_label,
        n_infer_steps=20,
    ):
        """Reverse the RF from a supplied scaled inversion endpoint.

        Unlike :meth:`sample_x0`, this method does not draw random noise.  It
        is used to compare an actual source inversion endpoint with its
        aligned counterpart under identical Brain-ID and target-label
        conditions.
        """
        if endpoint.ndim != 5:
            raise ValueError(
                "endpoint must have shape [B,C,H,W,D], got "
                f"{tuple(endpoint.shape)}"
            )
        if n_infer_steps < 1:
            raise ValueError(
                f"n_infer_steps must be >= 1, got {n_infer_steps}"
            )
        device = endpoint.device
        batch = endpoint.shape[0]
        if isinstance(class_label, int):
            class_label = torch.full(
                (batch,),
                class_label,
                device=device,
                dtype=torch.long,
            )
        else:
            class_label = class_label.to(device=device, dtype=torch.long)

        timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            n_infer_steps + 1,
            device=device,
        )
        timesteps = timestep_transform(
            timesteps,
            int(np.prod(endpoint.shape[2:])),
            base_img_size_numel=32768,
            scale=1.4,
            num_train_timesteps=self.num_train_timesteps,
            spatial_dim=3,
        )
        state = endpoint
        for index in range(n_infer_steps):
            current_t = timesteps[index].expand(batch)
            velocity = self.forward(
                state,
                current_t,
                class_label,
                feats,
            )
            delta = (
                timesteps[index] - timesteps[index + 1]
            ) / self.num_train_timesteps
            state = state + velocity * delta
        return state / self.scale_factor

    # ── Training-mode generation (for Stage 2 adversarial) ── #

    def generate_train(self, feats, class_label, z_src, source_dropout_p=0.1,
                       t=None, n_infer_steps=1, return_diagnostics=False):
        """Differentiable Stage-2 graft sample in scaled latent space.

        Legacy ``concat`` mode retains the random-t one-forward estimator for
        reproducibility. ``adapter`` mode mirrors deployment: it starts at pure
        noise and performs differentiable Euler steps, while z_src enters only
        through the adapter. This removes the old dual use of z_src as both RF
        starting point and direct condition.
        """
        B = feats[0].shape[0]
        device = feats[0].device
        z_src_scaled = z_src.to(device).float() * self.scale_factor

        z_src_cond = z_src_scaled
        keep = torch.ones((B,), device=device, dtype=z_src_scaled.dtype)
        if source_dropout_p > 0:
            keep = (torch.rand((B,), device=device) >= source_dropout_p).to(z_src_scaled.dtype)
            z_src_cond = z_src_cond * keep.view(-1, 1, 1, 1, 1)

        if self.source_injection == "adapter":
            if n_infer_steps < 1:
                raise ValueError(f"n_infer_steps must be >=1, got {n_infer_steps}")
            spatial_numel = int(np.prod(z_src_scaled.shape[2:]))
            timesteps = torch.linspace(
                self.num_train_timesteps - 1, 0, n_infer_steps + 1, device=device
            )
            timesteps = timestep_transform(
                timesteps, spatial_numel,
                base_img_size_numel=32768,
                scale=1.4,
                num_train_timesteps=self.num_train_timesteps,
                spatial_dim=3,
            )
            z_t = torch.randn_like(z_src_scaled)
            for i in range(n_infer_steps):
                t_batch = timesteps[i].expand(B)
                v_pred = self.forward(
                    z_t, t_batch, class_label, feats, z_src=z_src_cond
                )
                dt = (timesteps[i] - timesteps[i + 1]) / self.num_train_timesteps
                z_t = z_t + v_pred * dt
            diagnostics = {
                "source_drop_fraction": (1.0 - keep).mean().detach(),
                "graft_start_source_fraction": z_t.new_zeros(()),
                "graft_steps": z_t.new_tensor(float(n_infer_steps)),
            }
            return (z_t, diagnostics) if return_diagnostics else z_t

        if t is None:
            t = torch.rand((B,), device=device) * self.num_train_timesteps
            spatial_numel = int(np.prod(z_src_scaled.shape[2:]))
            t = timestep_transform(
                t, spatial_numel,
                base_img_size_numel=32768,
                scale=1.4,
                num_train_timesteps=self.num_train_timesteps,
                spatial_dim=3,
            )
        else:
            t = t.to(device=device, dtype=torch.float)

        alpha = 1.0 - t / self.num_train_timesteps
        alpha_view = alpha.view(-1, 1, 1, 1, 1)
        noise = torch.randn_like(z_src_scaled)
        z_t = alpha_view * z_src_scaled + (1.0 - alpha_view) * noise

        v_pred = self.forward(z_t, t, class_label, feats, z_src=z_src_cond)
        t_view = (t / self.num_train_timesteps).view(-1, 1, 1, 1, 1)
        x0_hat = z_t + t_view * v_pred
        diagnostics = {
            "source_drop_fraction": (1.0 - keep).mean().detach(),
            "graft_start_source_fraction": alpha.mean().detach(),
            "graft_steps": x0_hat.new_ones(()),
        }
        return (x0_hat, diagnostics) if return_diagnostics else x0_hat

    # ── Checkpoint loading ── #

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
        """Load pretrained UNet (4ch conv_in intact), attach fresh ControlNet.

        ControlNet is randomly initialized (DO NOT copy UNet weights).
        This avoids the oscillation observed in V5 when ControlNet was
        initialized with copied UNet weights while the UNet was trainable.
        """
        bridge = BridgeRFV7(
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
        )

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle Stage 1/2 training checkpoint format (full bridge state dict)
        if "bridge" in sd:
            raw_state = BridgeRFV7._adapt_source_channels(sd["bridge"], bridge)
            BridgeRFV7._load_bridge_state(bridge, raw_state)
            # Restore scale_factor from checkpoint (was missing — caused
            # vis/inference scripts that use from_pretrained directly to
            # silently use scale_factor=1.0, producing garbage VAE decodes).
            if "scale_factor" in sd:
                bridge.scale_factor = float(sd["scale_factor"])
                print(f"[V7] Restored scale_factor={bridge.scale_factor:.6f} from checkpoint")
            if "num_train_timesteps" in sd:
                bridge.num_train_timesteps = int(sd["num_train_timesteps"])
            if load_ema and "ema" in sd:
                ema_state = BridgeRFV7._adapt_source_channels(sd["ema"], bridge)
                result = bridge.load_state_dict(ema_state, strict=False)
                float_keys = {k for k, v in bridge.state_dict().items()
                              if v.dtype.is_floating_point}
                missing_float = [
                    k for k in result.missing_keys
                    if k in float_keys and not k.startswith("source_adapter.")
                ]
                if missing_float:
                    raise RuntimeError(
                        f"[V7] EMA checkpoint missing {len(missing_float)} float keys "
                        f"(first 5: {missing_float[:5]}). "
                        f"Checkpoint/model architecture mismatch."
                    )
                print(f"[V7] Loaded EMA weights for inference/vis")
            else:
                print(f"[V7] Loaded full bridge from training checkpoint")
            bridge = bridge.to(device)
            return bridge

        # Handle raw pretrained format
        if "state_dict" in sd:
            sd = sd["state_dict"]
        if "unet_state_dict" in sd:
            sd = sd["unet_state_dict"]

        # Strip DDP prefix
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

        # Pretrained has 128 classes; we use 15 -> reinit class_embedding
        cleaned.pop("class_embedding.weight", None)

        # conv_in stays 4ch (pretrained intact)
        missing, unexpected = bridge.unet.load_state_dict(cleaned, strict=False)
        missing_real = [k for k in missing if "class_embedding" not in k]
        if missing_real:
            print(f"[V7] UNet missing (first 5): {missing_real[:5]}")
        if unexpected:
            print(f"[V7] UNet unexpected (first 5): {unexpected[:5]}")

        # ControlNet: random init (no copy from UNet)
        bridge = bridge.to(device)
        print(f"[V7] Bridge ready: in_channels={bridge.in_channels}, "
              f"num_train_timesteps={num_train_timesteps}, "
              f"controlnet=on (feat4 {feat4_channels}ch, "
              f"conditioning_embedding="
              f"{list(bridge.conditioning_embedding_num_channels)}, random init), "
              f"Euler ODE inference")
        return bridge

    @staticmethod
    def _load_bridge_state(bridge, state_dict):
        """Load V7 state, permitting only a new zero-init source adapter."""
        result = bridge.load_state_dict(state_dict, strict=False)
        missing = [
            k for k in result.missing_keys
            if not k.startswith("source_adapter.")
        ]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                "[V7] Bridge checkpoint mismatch: "
                f"missing={missing[:5]}, unexpected={result.unexpected_keys[:5]}"
            )
        if result.missing_keys:
            print("[V7b-adapter] Source adapter absent in checkpoint; kept zero initialized")

    @staticmethod
    def _adapt_source_channels(state_dict, bridge):
        """Zero-expand a V7a UNet conv_in state for fair V7b continuation."""
        state_dict = dict(state_dict)
        key = "unet.conv_in.conv.weight"
        if key not in state_dict:
            return state_dict
        old_weight = state_dict[key]
        new_weight = bridge.state_dict()[key]
        if old_weight.shape == new_weight.shape:
            return state_dict
        if (old_weight.shape[0] == new_weight.shape[0]
                and old_weight.shape[1] == bridge.z_channels
                and new_weight.shape[1] == bridge.z_channels * 2
                and old_weight.shape[2:] == new_weight.shape[2:]):
            expanded = torch.zeros_like(new_weight, device=old_weight.device)
            expanded[:, :bridge.z_channels].copy_(old_weight)
            state_dict[key] = expanded
            print("[V7b] Expanded UNet conv_in 4->8; z_src weights zero-initialized")
            return state_dict
        raise RuntimeError(
            f"Cannot adapt {key} from {tuple(old_weight.shape)} to {tuple(new_weight.shape)}"
        )


# ── Training loss (RF forward) ── #

def spatial_axis_overlap_weights(
    axis_length,
    image_slice_range,
    image_axis_size,
    *,
    device=None,
    dtype=torch.float32,
):
    """Return exact image-to-latent overlap weights for one spatial axis.

    A latent cell represents a continuous interval in the preprocessed image.
    Boundary cells therefore receive fractional weights when an image-space
    evaluation slab is not divisible by the VAE downsampling factor.  This is
    preferable to rounding the 30-slice challenge slab to either 28 or 32
    supervised image slices.
    """
    axis_length = int(axis_length)
    image_axis_size = int(image_axis_size)
    if axis_length < 1 or image_axis_size < 1:
        raise ValueError("axis sizes must be positive")
    if len(image_slice_range) != 2:
        raise ValueError("image_slice_range must contain [start, end]")
    start, end = (float(value) for value in image_slice_range)
    if not 0.0 <= start < end <= float(image_axis_size):
        raise ValueError(
            "image_slice_range must lie inside the preprocessed image axis, "
            f"got [{start}, {end}) for size {image_axis_size}"
        )

    cell_width = float(image_axis_size) / axis_length
    cell_start = torch.arange(
        axis_length, device=device, dtype=torch.float32
    ) * cell_width
    cell_end = cell_start + cell_width
    overlap = (
        torch.minimum(cell_end, torch.tensor(end, device=device))
        - torch.maximum(cell_start, torch.tensor(start, device=device))
    ).clamp_min(0.0)
    weights = overlap / cell_width
    if not bool(weights.gt(0).any().item()):
        raise ValueError("image_slice_range does not overlap the latent axis")
    return weights.to(dtype=dtype)


def _weighted_axis_l1(prediction, target, weights, axis=-1):
    """L1 mean over one softly selected spatial slab."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != target shape {target.shape}"
        )
    axis = int(axis) % prediction.ndim
    weights = torch.as_tensor(
        weights, device=prediction.device, dtype=torch.float32
    )
    if weights.ndim != 1 or weights.numel() != prediction.shape[axis]:
        raise ValueError(
            "spatial weights must be one-dimensional and match the selected "
            f"axis ({prediction.shape[axis]}), got {tuple(weights.shape)}"
        )
    if not bool(weights.ge(0).all().item()) or weights.sum().item() <= 0:
        raise ValueError("spatial weights must be non-negative with positive sum")

    view_shape = [1] * prediction.ndim
    view_shape[axis] = weights.numel()
    error = (prediction - target).abs().float()
    weighted_error = error * weights.view(view_shape)
    values_per_axis_cell = error.numel() / error.shape[axis]
    return weighted_error.sum() / (weights.sum() * values_per_axis_cell)

def rf_loss_v7(
    bridge,
    z_0,
    feats,
    class_label,
    scale_factor=1.0,
    t=None,
    z_src=None,
    endpoint=None,
    endpoint_focus_probability=0.0,
    endpoint_min_fraction=0.7,
    spatial_axis_weights=None,
    spatial_axis=-1,
):
    """Rectified Flow loss: L1(v_pred, v_target).

    Args:
        bridge: BridgeRFV7.
        z_0: (B, 4, 96³) clean target latent (unscaled).
        feats: list of 5 Brain-ID features.
        class_label: (B,) target class (0-14).
        scale_factor: latent scaling factor.
        t: optional pre-sampled timesteps (None = sample with transform).
        z_src: optional UNSCALED source condition. The standard Stage-1 anchor
            keeps this None; adapter Stage 2 also uses an active-source anchor.
        endpoint: optional externally supplied *scaled* RF endpoint.

        spatial_axis_weights: optional soft selection over one latent spatial
            axis. ``None`` preserves the historical full-volume objective.
        spatial_axis: tensor axis matched by ``spatial_axis_weights``.

    Returns:
        loss: L1 loss (scalar).
    """
    z_0_scaled = z_0 * scale_factor
    z_src_scaled = None if z_src is None else z_src * scale_factor
    prediction, target, loss = bridge.forward_rf(
        z_0_scaled,
        feats,
        class_label,
        t=t,
        z_src=z_src_scaled,
        endpoint=endpoint,
        endpoint_focus_probability=endpoint_focus_probability,
        endpoint_min_fraction=endpoint_min_fraction,
    )
    if spatial_axis_weights is None:
        return loss
    return _weighted_axis_l1(
        prediction,
        target,
        spatial_axis_weights,
        axis=spatial_axis,
    )


# ── Diagnostic: ControlNet zero_conv norms ── #

def controlnet_norms(bridge):
    """Compact per-scale zero_conv weight norm (injection-strength proxy).

    Returns a one-line string suitable for epoch logging.
    """
    cn = bridge.controlnet
    parts = []
    for k in range(4):
        idx_start = k * 3
        ns = []
        for j in range(3):
            block = cn.controlnet_down_blocks[idx_start + j]
            w = block.conv.weight if hasattr(block, 'conv') else block.weight
            ns.append(w.norm().item())
        parts.append(f"s{k}=[{ns[0]:.3f},{ns[1]:.3f},{ns[2]:.3f}]")
    mid_w = cn.controlnet_mid_block.conv.weight if hasattr(cn.controlnet_mid_block, 'conv') else cn.controlnet_mid_block.weight
    parts.append(f"mid=[{mid_w.norm().item():.3f}]")
    return " cn=" + " ".join(parts)


def source_channel_norms(bridge):
    """Report source-path norms for concat or adapter shortcut monitoring."""
    if not bridge.use_source_latent:
        return ""
    if bridge.source_injection == "adapter":
        first = bridge.source_adapter.net[0].weight.norm().item()
        zero = bridge.source_adapter.net[-1].weight.norm().item()
        base = bridge.unet.conv_in.conv.weight.norm().item()
        ratio = zero / max(base, 1e-12)
        return (
            f" src_adapter=[first={first:.4f},out={zero:.4f},"
            f"out/base={ratio:.4f}]"
        )
    weight = bridge.unet.conv_in.conv.weight
    base_norm = weight[:, :bridge.z_channels].norm().item()
    src_norm = weight[:, bridge.z_channels:].norm().item()
    ratio = src_norm / max(base_norm, 1e-12)
    return f" src_conv=[base={base_norm:.4f},src={src_norm:.4f},ratio={ratio:.4f}]"
