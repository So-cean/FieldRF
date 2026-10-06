"""FieldRF-V6: SelfRDB Bridge + ControlNet (feat4) + learnable MLP endpoint.

V5 = V3 framework (proven) + clean schedule:
  - 4ch UNet input [z_t]  (s NOT fed to UNet — avoids identity shortcut)
  - Single forward training    (V3-style, no recursion)
  - DDIM inference             (V3 deterministic, no ancestral noise)
  - ControlNet 192^3           (feat4 64ch conditioning, 13-point residual)
  - prepend-0 + double-count fix (clean schedule, std/s = gamma exact)
  - q_posterior uses self.std  (correct for gamma=0.1)

Design (SelfRDB soft-prior bridge, not RF):
  Source: s = endpoint_proj(downsample(feat4)) -> 4ch (bridge endpoint = "skeleton")
  Bridge: z_t = mu_x0(t)*z_0 + mu_y(t)*s + std(t)*noise  (soft prior)
  Target: x0 prediction (NOT velocity), single forward pass
  Conditioning:
    - ControlNet: z_t (4ch) main input, feat4 (64ch@192^3) as conditioning
      -> 13 skip+mid residuals (z_t-dependent, detail injection)
    - UNet input: z_t (4ch) only — s lives only in the schedule, NOT in UNet
      (feeding s to UNet creates an identity shortcut that caps cross-field
      quality; V3 reaches 0.94 with s-only-in-schedule, so we match that)

  UNet: DiffusionModelUNetMaisi (conv_in 4ch, pretrained intact)
  ControlNet: monai ControlNet (4ch input, feat4 conditioning, zero-conv exits)
  Class: MAISI built-in class_embedding (15-class)
  Inference: DDIM sample_x0 (deterministic reverse, n_infer_steps subsampling)

SelfRDB schedule (from icon-lab/SelfRDB, verbatim port via maisi):
  betas = tent-shaped (symmetric, peak at midpoint)
  s = cumsum(betas)^0.5, s_bar = flip(cumsum(betas))^0.5
  mu_x0, mu_y = gaussian_product(s, s_bar)
  std = gamma * s  (double-count fix: s_bar^2 = total - s^2, so std/s = gamma exact)
  n_steps=100, gamma=0.1, beta_start=1e-4, beta_end=0.5
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.nets import ControlNet

from monai.utils.type_conversion import convert_to_tensor

# ── Class label mapping (same as V3) ── #
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
NUM_CLASSES = len(MODALITIES) * len(FIELDS)  # 15


def get_class_label(modality, field):
    return MODALITIES.index(modality) * len(FIELDS) + FIELDS.index(field)


# ── Grad-safe UNet wrapper (from V3, needed for ControlNet residual injection) ── #

class DiffusionModelUNetMaisiGradSafe(DiffusionModelUNetMaisi):
    """Out-of-place adds for down_block/mid_block residuals (autograd-safe).

    The upstream MAISI UNet uses in-place ``h += residual`` which breaks
    autograd when the residual has gradient (e.g. our ControlNet projections).
    This override replaces them with out-of-place ``h = h + residual``.
    """

    def _apply_down_blocks(self, h, emb, context, down_block_additional_residuals):
        from monai.utils.type_conversion import convert_to_tensor

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
                mid_block_additional_residual=None, **kwargs):
        h = self.conv_in(x)
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, context, None, spacing_tensor)
        h, _updated = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        # OUT-OF-PLACE add (was in-place upstream: h += mid_block_additional_residual).
        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual
        h = self._apply_up_blocks(h, emb, context, _updated)
        h = self.out(h)
        return convert_to_tensor(h)


# ── V5 Bridge: V3 framework (single forward + DDIM) + V4 architecture ── #

class BridgeRFV6(nn.Module):
    """FieldRF-V6: Bridge with ControlNet + endpoint input, DDIM inference.

    V3 framework: single forward training, DDIM inference (no ancestral noise).
    V4 architecture: ControlNet 192³, endpoint as UNet input, clean schedule.

    UNet input: concat[z_t(4) | s(4)] = 8ch
      - z_t: current bridge sample (from q_sample)
      - s: source projection (bridge endpoint, anatomical prior)

    No R=2 recursion (V3-style single forward). No ancestral posterior in
    inference (DDIM deterministic). Simpler, faster convergence, no cold start
    from s zero-init (endpoint_proj).

    Source: feat4 (192³) → feat4_downsample → 96³ → endpoint_proj (learnable MLP, zero-init).
    Schedule: SelfRDB bridge (gamma=0.1, prepend-0, double-count fixed).
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
        # Latent
        z_channels=4,
        # SelfRDB
        n_steps=10,
        gamma=1.0,
        beta_start=1e-4,
        beta_end=3.0,
        scale_factor=1.0,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.feat4_channels = feat4_channels
        self.scale_factor = scale_factor

        # ── UNet (4ch conv_in = z_t only, pretrained intact) ── #
        # s (endpoint) is NOT fed to UNet — only participates in the bridge
        # schedule (q_sample / _ddim_step). This matches V3's controlnet
        # injection mode and avoids the identity shortcut where UNet relies
        # on s instead of learning to denoise z_t. feat4 still enters via
        # ControlNet (13-point residual injection, zero_conv-gated).
        self.in_channels = z_channels  # 4 (z_t only)
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
            use_flash_attention=True,
        )

        # ── ControlNet (4ch input = z_t, feat4 conditioning) ── #
        # feat4 at 192³ → conditioning_embedding [16,64] → 64ch@96³ (1 downsample).
        # 16ch intermediate is a bottleneck but V3 reaches 0.94 with it, so keep
        # [16,64] for a clean ablation (isolate the s-input variable).
        self.controlnet = ControlNet(
            spatial_dims=3,
            in_channels=self.in_channels,  # 4ch (z_t only, same as UNet)
            num_res_blocks=list(num_res_blocks),
            channels=list(unet_channels),
            attention_levels=list(attention_levels),
            norm_num_groups=norm_num_groups,
            num_head_channels=list(num_head_channels),
            use_flash_attention=True,
            num_class_embeds=num_class_embeds,
            conditioning_embedding_in_channels=feat4_channels,  # 64
            conditioning_embedding_num_channels=[16, 64],  # 192³→96³ via 1 stride-2
        )

        # ── Source projection: learnable MLP (V3-style, zero-init) ── #
        # feat4_downsample: 192³→96³ (learnable, avg_pool init — field-agnostic)
        self.feat4_downsample = nn.Conv3d(
            feat4_channels, feat4_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        nn.init.constant_(self.feat4_downsample.weight, 1.0 / 27)
        self.register_buffer("feat4_downsample_init", self.feat4_downsample.weight.data.clone())

        # endpoint_proj: feat4_ds 64ch → 4ch endpoint (learnable, zero-init last layer)
        self.endpoint_proj = nn.Sequential(
            nn.Conv3d(feat4_channels, 16, 1),
            nn.GroupNorm(8, 16),
            nn.GELU(),
            nn.Conv3d(16, z_channels, 1),
        )
        nn.init.zeros_(self.endpoint_proj[-1].weight)
        nn.init.zeros_(self.endpoint_proj[-1].bias)

        # Fixed spacing tensor (field info in class label, not spacing)
        self.register_buffer(
            "spacing_tensor", torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        )

        # ── SelfRDB schedule ── #
        self.n_steps = n_steps
        self.gamma = gamma
        self._setup_schedule(n_steps, gamma, beta_start, beta_end)

    # ── SelfRDB schedule (from selfrdb_bridge.py, verbatim) ── #

    @staticmethod
    def _get_betas(n_steps, beta_start, beta_end):
        """Tent-shaped (symmetric) beta schedule — V3-compatible (correct peak).

        V5's version had a bug: linspace from beta_start + prepend-0 + flip
        discarded the upper half of the linspace, so the tent peaked at the
        linspace midpoint (~0.13-0.26) instead of beta_end (0.5). With
        beta_start=0.1 it coincidentally matched V3's total (~17); with
        beta_start=1e-4 the bug was exposed (total=4, std_peak=0.20 not 0.41).

        Fix: use V3's construction — ascending from beta_start^0.5 to
        beta_end^0.5 over half+1 points, then mirror. Peak = beta_end exactly.
        No prepend-0 (t=0 has tiny noise std≈0.001, negligible vs 0).
        """
        half = n_steps // 2
        ascending = np.linspace(beta_start ** 0.5, beta_end ** 0.5, half + 1) ** 2
        if n_steps % 2 == 0:
            betas = np.concatenate([ascending, np.flip(ascending[:-1])])
        else:
            betas = np.concatenate([ascending, np.flip(ascending)])
        return betas[: n_steps + 1].astype(np.float32)

    @staticmethod
    def _gaussian_product(sigma1, sigma2, denom=None):
        """Product of two Gaussians: returns (mu1, mu2, var).
        
        denom: if provided, used as the denominator instead of sigma1²+sigma2².
        """
        if denom is None:
            denom = sigma1 ** 2 + sigma2 ** 2
        return sigma2 ** 2 / denom, sigma1 ** 2 / denom, (sigma1 ** 2 * sigma2 ** 2) / denom

    def _setup_schedule(self, n_steps, gamma, beta_start, beta_end):
        betas = self._get_betas(n_steps, beta_start, beta_end)
        total = betas.sum()
        # Fix discrete double-counting: official uses s=cumsum(betas)^0.5 (inclusive)
        # and s_bar=flip(cumsum(betas))^0.5 (also inclusive of betas[t]). This makes
        # s²+s̄² = total + betas[t] (betas[t] double-counted), not total (constant).
        # Paper's continuous integral: σ²+σ̄² = ∫β dτ (constant). 
        # Fix: s̄² = total - s² (exclusive of betas[t]) → s²+s̄² = total exactly.
        s_sq = np.cumsum(betas)          # s²(t) = betas[0]+...+betas[t] (inclusive)
        s_bar_sq = total - s_sq           # s̄²(t) = total - s²(t) (exclusive → no double count)
        s = s_sq ** 0.5
        s_bar = s_bar_sq ** 0.5
        mu_x0, mu_y, _ = self._gaussian_product(s, s_bar, denom=total)
        gamma_scaled = gamma * total
        std = gamma_scaled * s / total

        self.register_buffer("s", torch.tensor(np.ascontiguousarray(s), dtype=torch.float32))
        self.register_buffer("mu_x0", torch.tensor(np.ascontiguousarray(mu_x0), dtype=torch.float32))
        self.register_buffer("mu_y", torch.tensor(np.ascontiguousarray(mu_y), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.ascontiguousarray(std), dtype=torch.float32))

    @staticmethod
    def _view(coef, t, ndim):
        shape = [-1] + [1] * (ndim - 1)
        return coef[t].view(shape)

    def q_sample(self, t, z_0, s):
        """Forward: x_t = mu_x0*z_0 + mu_y*s + std*noise (soft prior)."""
        nd = z_0.ndim
        return (self._view(self.mu_x0, t, nd) * z_0
                + self._view(self.mu_y, t, nd) * s
                + self._view(self.std, t, nd) * torch.randn_like(z_0))

    def q_posterior(self, t, z_t, z_0_pred, s, t_prev=None):
        """Reverse: sample x_{t_prev} ~ p(x_{t_prev} | x_t, x0, s) — ancestral.

        NOTE: V6 UNUSED (dead code). Kept for reference / potential Stage-2
        ancestral ablation. V5 inference uses DDIM (_ddim_step via sample_x0),
        which is deterministic and matches V3's proven framework.

        Uses self.std (gamma-scaled noise) — consistent with q_sample.
        
        NOTE: Official SelfRDB uses self.s (cumulative sqrt betas). But official
        uses gamma=1.0 where s/std ratio ≈ 1.1x (nearly equal). Our gamma=0.1
        makes s/std = 10.2x — using self.s injects 408% signal-magnitude noise
        in the posterior, destroying z_t. With gamma=0.1, self.std is the correct
        choice: it matches the actual forward noise level.
        """
        nd = z_0_pred.ndim
        if t_prev is None:
            t_prev = t - 1
        std_t = self._view(self.std, t, nd)
        std_prev = self._view(self.std, t_prev, nd)
        mu_x0_t = self._view(self.mu_x0, t, nd)
        mu_x0_prev = self._view(self.mu_x0, t_prev, nd)
        mu_y_t = self._view(self.mu_y, t, nd)
        mu_y_prev = self._view(self.mu_y, t_prev, nd)

        var_t = std_t ** 2
        var_prev = std_prev ** 2
        var_t_prev = var_t - var_prev * (mu_x0_t / mu_x0_prev.clamp(min=1e-8)) ** 2
        v = var_t_prev * (var_prev / var_t)
        z_prev_mean = (
            mu_x0_prev * z_0_pred + mu_y_prev * s
            + ((var_prev - v) / var_t).clamp_min(0).sqrt()
            * (z_t - mu_x0_t * z_0_pred - mu_y_t * s)
        )
        return z_prev_mean + v.clamp_min(0).sqrt() * torch.randn_like(z_t)

    # ── Source ── #

    def get_source(self, feats):
        """Compute bridge source s from feat4 via learnable downsample + MLP.
        
        feat4 (192³) → feat4_downsample → 96³ → endpoint_proj → s (4ch, 96³)
        """
        feat4 = feats[-1].float()
        feat4_ds = self.feat4_downsample(feat4)
        return self.endpoint_proj(feat4_ds)

    # ── Forward (single pass, no recursion) ── #

    def _unet_forward(self, z_t, s, timesteps, class_label, feat4):
        """Single UNet forward: z_t (4ch) + ControlNet(feat4).

        s is NOT fed to UNet/ControlNet — it only lives in the bridge schedule.
        This avoids the identity shortcut (UNet relying on s instead of
        denoising z_t). feat4 enters via ControlNet (13-point residual injection).
        """
        # ControlNet: z_t as main input, feat4 as conditioning
        down_res, mid_res = self.controlnet(
            z_t, timesteps=timesteps,
            controlnet_cond=feat4, class_labels=class_label,
        )

        B = z_t.shape[0]
        return self.unet(
            z_t, timesteps=timesteps,
            class_labels=class_label,
            spacing_tensor=self.spacing_tensor.expand(B, -1),
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )

    def forward(self, z_t, t, class_label, feats):
        """Predict x0 from (z_t, t, feats) — single forward pass (V3-style).

        Args:
            z_t: (B, 4, 96³) bridge sample (from q_sample, SCALED by scale_factor).
            t: (B,) SelfRDB timestep indices [0, n_steps].
            class_label: (B,) target class (0-14).
            feats: list of 5 Brain-ID features [feat0..feat4].

        Returns:
            x0_pred: (B, 4, 96³) predicted target latent (SCALED).
        """
        s = self.get_source(feats) * self.scale_factor  # scaled endpoint
        feat4 = feats[-1].float()   # (B, 64, 192³) — ControlNet conditioning

        # Map SelfRDB timestep to UNet timestep
        t_mapped = torch.round(t.float() / self.n_steps * 999).long().clamp(0, 999)

        return self._unet_forward(z_t, s, t_mapped, class_label, feat4)

    def generate_train(self, feats, class_label):
        """Generate x0 prediction for adversarial training (WITH gradient).

        Single forward pass (no R=2 recursion).
        Used by Stage 2 graft: z_fake = generate_train(graft_feats, graft_class).
        """
        B = feats[0].shape[0]
        device = feats[0].device

        s = self.get_source(feats) * self.scale_factor  # SCALED endpoint
        t = torch.randint(1, self.n_steps + 1, (B,), device=device)
        z_t = self.q_sample(t, torch.zeros_like(s), s).detach()

        return self.forward(z_t, t, class_label, feats)  # single forward

    # ── Inference ── #

    def _ddim_step(self, t, t_prev, z_t, z_0_pred, s):
        """DDIM deterministic step: z_t → z_{t_prev}.

        Uses self.std (consistent with q_sample). Deterministic (no noise injection).
        Valid for arbitrary (t, t_prev) pairs — supports subsampled inference.
        """
        nd = z_0_pred.ndim
        mu_x0_t = self._view(self.mu_x0, t, nd)
        mu_x0_prev = self._view(self.mu_x0, t_prev, nd)
        mu_y_t = self._view(self.mu_y, t, nd)
        mu_y_prev = self._view(self.mu_y, t_prev, nd)
        std_t = self._view(self.std, t, nd).clamp(min=1e-8)
        std_prev = self._view(self.std, t_prev, nd)

        eps = (z_t - mu_x0_t * z_0_pred - mu_y_t * s) / std_t
        return mu_x0_prev * z_0_pred + mu_y_prev * s + std_prev * eps

    @torch.no_grad()
    def sample_x0(self, feats, class_label, num_infer_steps=20, device=None):
        """DDIM inference: deterministic reverse from endpoint to target.

        No ancestral noise (deterministic). Single forward per step (no R=2).
        Subsampled: num_infer_steps < n_steps is OK (like DDPM→DDIM).

        Args:
            feats: list of 5 Brain-ID features.
            class_label: (B,) or int target class.
            num_infer_steps: number of DDIM steps (default 20).
        Returns:
            x0_pred: (B, 4, 96³) predicted target latent (UNSCCALED, for VAE decode).
        """
        if device is None:
            device = next(self.parameters()).device

        B = feats[0].shape[0]
        if isinstance(class_label, int):
            class_label = torch.full((B,), class_label, device=device, dtype=torch.long)
        else:
            class_label = class_label.to(device)

        s = self.get_source(feats).to(device) * self.scale_factor
        feat4 = feats[-1].float().to(device)

        if num_infer_steps is None or num_infer_steps > self.n_steps:
            num_infer_steps = min(num_infer_steps or self.n_steps, self.n_steps)

        step_indices = np.linspace(0, self.n_steps, num_infer_steps + 1, dtype=int)
        timesteps = list(reversed(step_indices.tolist()))

        # Start: z_T = endpoint + noise (soft prior, SCALED)
        t_T = torch.full((B,), self.n_steps, device=device, dtype=torch.long)
        z_t = self.q_sample(t_T, torch.zeros_like(s), s)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Single forward: predict x0 (no R=2 recursion)
            z_0_pred = self.forward(z_t, t_batch, class_label, feats)

            # DDIM step (deterministic, no noise)
            t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
            z_t = self._ddim_step(t_batch, t_prev_batch, z_t, z_0_pred, s)

        # Final prediction at last timestep
        z_0_pred = self.forward(z_t, timesteps[-1] if isinstance(timesteps[-1], torch.Tensor)
                                else torch.full((B,), timesteps[-1], device=device, dtype=torch.long),
                                class_label, feats)

        return z_0_pred / self.scale_factor

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
        z_channels=4,
        n_steps=100,
        gamma=0.1,
        beta_start=1e-4,
        beta_end=0.5,
        scale_factor=1.0,
        device="cpu",
        load_ema=False,
        **kwargs,
    ):
        """Load pretrained UNet (4ch conv_in intact), attach ControlNet.

        No conv_in expansion — UNet keeps pretrained 4ch (z_t only). s is not
        a UNet input (avoids identity shortcut). feat4 enters via ControlNet.

        Args:
            ckpt_path: MAISI pretrained UNet checkpoint.
        """
        bridge = BridgeRFV6(
            unet_channels=unet_channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            num_class_embeds=num_class_embeds,
            feat4_channels=feat4_channels,
            z_channels=z_channels,
            n_steps=n_steps,
            gamma=gamma,
            beta_start=beta_start,
            beta_end=beta_end,
            scale_factor=scale_factor,
        )

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle Stage 1/2 training checkpoint format (full bridge state dict)
        if "bridge" in sd:
            bridge.load_state_dict(sd["bridge"])
            if load_ema and "ema" in sd:
                result = bridge.load_state_dict(sd["ema"], strict=False)
                float_keys = {k for k, v in bridge.state_dict().items()
                              if v.dtype.is_floating_point}
                missing_float = [k for k in result.missing_keys if k in float_keys]
                if missing_float:
                    raise RuntimeError(
                        f"[V6] EMA checkpoint missing {len(missing_float)} float keys "
                        f"(first 5: {missing_float[:5]}). "
                        f"Checkpoint/model architecture mismatch."
                    )
                print(f"[V6] Loaded EMA weights for inference/vis")
            else:
                print(f"[V6] Loaded full bridge from training checkpoint")
            bridge = bridge.to(device)
            return bridge

        # Handle V3-compatible format
        if "state_dict" in sd:
            sd = sd["state_dict"]
        if "unet_state_dict" in sd:
            sd = sd["unet_state_dict"]

        # Strip DDP prefix
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

        # Pretrained has 128 classes; we use 15 -> reinit class_embedding
        cleaned.pop("class_embedding.weight", None)

        # conv_in stays 4ch (pretrained intact) — no expansion needed.
        # UNet input = z_t only; s is not a UNet input.

        # Load UNet weights (strict=False: skip missing/extra keys)
        missing, unexpected = bridge.unet.load_state_dict(cleaned, strict=False)
        missing_real = [k for k in missing if "class_embedding" not in k]
        if missing_real:
            print(f"[V6] UNet missing (first 5): {missing_real[:5]}")
        if unexpected:
            print(f"[V6] UNet unexpected (first 5): {unexpected[:5]}")

        # ── ControlNet: random init (V3-style, NOT copied from UNet) ── #
        # V6 (was V5): copied UNet down_blocks into ControlNet -> injected
        # strong pretrained features once zero_convs grew -> over-specialization
        # -> oscillation after ep25 (peaked 0.918, dropped to 0.912).
        # V3 uses fresh random ControlNet (no copy) -> weak->gradual injection
        # acts as implicit regularization -> monotonic rise to 0.94.
        # Match V3: leave ControlNet at default random init.

        bridge = bridge.to(device)
        print(f"[V6] Bridge ready: in_channels={bridge.in_channels}, "
              f"n_steps={n_steps}, gamma={gamma}, "
              f"controlnet=on (feat4 {feat4_channels}ch@192³, random init), "
              f"DDIM inference")
        return bridge


# ── Training Loss (single forward, no R=2) ── #

def selfrdb_loss_v6(bridge, z_0, feats, class_label, scale_factor=1.0):
    """Bridge training loss: single forward + L1 (V3-style, no recursion).

    Args:
        bridge: BridgeRFV6
        z_0: (B, 4, 96³) clean target latent (unscaled).
        feats: list of 5 Brain-ID features [feat0, ..., feat4].
        class_label: (B,) target class (0-14).
        scale_factor: latent scaling factor.

    Returns:
        loss: L1(z_0_pred, z_0) single forward (on SCALED latents).
    """
    B = z_0.shape[0]
    device = z_0.device

    z_0_scaled = z_0 * scale_factor
    s = bridge.get_source(feats) * scale_factor  # scaled endpoint

    # Sample bridge timestep [0, n_steps]
    t = torch.randint(0, bridge.n_steps + 1, (B,), device=device)

    # Forward: z_t = mu_x0*z_0 + mu_y*s + std*noise (detached)
    z_t = bridge.q_sample(t, z_0_scaled, s).detach()

    # Single forward: predict x0 (no R=2 recursion)
    z_0_pred = bridge.forward(z_t, t, class_label, feats)

    # L1 loss (mean reduction — 3D latents have 3.5M elements)
    return F.l1_loss(z_0_pred, z_0_scaled, reduction="mean")
