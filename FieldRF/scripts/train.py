"""FieldRF Training: RF identity pretraining and source-conditioned RF latent graft finetuning.

Stage 1 (adv.enabled=False):
  loss = rectified_flow_loss(bridge, z_0, feats, class_label, scale_factor)
  (Rectified Flow: z_t = (1-t/T)*z_0 + (t/T)*noise, L1 on velocity)

Stage 2 supports two source paths:
  - concat: legacy source-conditioned RF [z_t | z_src] and random-t one-forward graft.
  - adapter: identity RF's 4ch UNet stays intact; a zero-init, time-gated source
    adapter is trained with active-source identity and pure-noise Euler grafts.

Key differences from RF:
  - Rectified Flow instead of SelfRDB bridge schedule.
  - scale_factor uses MAISI pretrained default (0.97045) directly from config.
  - Timestep-transform (MAISI recipe, scale=1.4, base=32³).
  - ControlNet conditioning_embedding [64,64] (was [16,64]).
  - No source endpoint projection (feat4_downsample, endpoint_proj removed).
  - Euler ODE inference (was DDIM).
"""

import argparse, os, sys, time, math, json
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import yaml

# FieldRF imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.model import (
    FieldRFModel, controlnet_norms, source_channel_norms,
    get_class_label, NUM_CLASSES, MODALITIES, timestep_transform,
)
from losses import rectified_flow_loss, spatial_axis_overlap_weights
from data.dataset import MultiFieldCacheDataset, PairedVisualizationDataset

try:
    import utils.npu_setup
except ImportError:
    pass


# ── EMA ── #

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        """Swap EMA weights into model. Returns backup dict for restore()."""
        backup = {}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd and sd[k].shape == v.shape:
                backup[k] = sd[k].clone()
                sd[k].copy_(v)
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        """Restore original weights from backup dict."""
        sd = model.state_dict()
        for k, v in backup.items():
            if k in sd:
                sd[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in list(self.shadow.keys()):
            if k in sd and sd[k].shape == self.shadow[k].shape:
                self.shadow[k].copy_(sd[k])


# ── Projection Discriminator (Stage 2 only, kept for future use) ── #

class LatentProjectionDiscriminator(nn.Module):
    """3D PatchGAN with projection conditioning."""

    def __init__(self, in_channels=4, num_classes=NUM_CLASSES, base_ch=64,
                 use_spectral_norm=False):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, base_ch * 4)

        def _sn(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        self.conv1 = nn.Sequential(
            _sn(nn.Conv3d(in_channels, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            _sn(nn.Conv3d(base_ch, base_ch * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm3d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            _sn(nn.Conv3d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm3d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = _sn(nn.Conv3d(base_ch * 4, 1, 4, padding=1))

    def forward(self, z, class_label):
        f1 = self.conv1(z)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        patch_logits = self.head(f3)
        feat_pooled = f3.mean(dim=[2, 3, 4])
        class_vec = self.class_emb(class_label)
        proj_logits = (feat_pooled * class_vec).sum(dim=1, keepdim=True)
        return patch_logits, proj_logits, [f1, f2, f3]

    def full_logits(self, z, class_label):
        patch, proj, _ = self.forward(z, class_label)
        return patch + proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def lsgan_d_loss(real_logits, fake_logits):
    real_target = torch.ones_like(real_logits)
    fake_target = torch.zeros_like(fake_logits)
    return 0.5 * (F.mse_loss(real_logits, real_target) +
                  F.mse_loss(fake_logits, fake_target))


def lsgan_g_loss(fake_logits):
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


def _grad_norm(parameters):
    """L2 norm of currently accumulated gradients."""
    total = None
    for p in parameters:
        if p.grad is None:
            continue
        value = p.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return 0.0 if total is None else total.sqrt().item()


@torch.no_grad()
def _latent_moment_gap(fake, real):
    """Channel-wise mean/std gaps; distribution diagnostic, not a loss."""
    dims = (0, 2, 3, 4)
    mean_gap = (fake.float().mean(dims) - real.float().mean(dims)).abs().mean()
    std_gap = (fake.float().std(dims) - real.float().std(dims)).abs().mean()
    return mean_gap.item(), std_gap.item()


# ── Main training function ── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--pretrain_ckpt", default=None)
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint (e.g. runs/pretrain_t1w/fieldrf_epoch50.pt)")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Override config train.n_epochs (for smoke testing)")
    parser.add_argument("--max_volumes", type=int, default=0,
                        help="Max volumes per field (0=all, for smoke testing)")
    args = parser.parse_args()

    # ── Load config ── #
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Distributed setup ── #
    if args.dist:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        rank, world_size, local_rank = 0, 1, 0

    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype = getattr(torch, cfg.get("train", {}).get("mixed_precision", "bfloat16"))

    data_cfg = cfg["data"]
    bridge_cfg = cfg["bridge"]
    train_cfg = cfg["train"]
    adv_cfg = train_cfg.get("adv", {})
    vis_cfg = cfg.get("vis", {})
    vae_cfg = cfg["vae"]
    inf_cfg = cfg.get("inference", {})

    spatial_loss_cfg = train_cfg.get("spatial_loss", {})
    spatial_loss_enabled = bool(spatial_loss_cfg.get("enabled", False))
    spatial_axis_weights = None
    spatial_crop_bounds = None
    spatial_axis = int(spatial_loss_cfg.get("axis", -1))
    if spatial_axis not in {-1, 4}:
        raise ValueError(
            "FieldRF evaluation-slab training currently supports only the "
            "axial/last latent axis (axis=-1 or 4)"
        )
    original_slice_range = tuple(
        spatial_loss_cfg.get("original_slice_range", [150, 180])
    )
    original_axis_size = int(spatial_loss_cfg.get("original_axis_size", 364))
    preprocessed_axis_size = int(
        spatial_loss_cfg.get("preprocessed_axis_size", 384)
    )
    if spatial_loss_enabled:
        if len(original_slice_range) != 2:
            raise ValueError("spatial_loss.original_slice_range needs 2 values")
        if original_axis_size <= preprocessed_axis_size:
            spatial_offset = (preprocessed_axis_size - original_axis_size) // 2
        else:
            spatial_offset = -((original_axis_size - preprocessed_axis_size) // 2)
        preprocessed_slice_range = tuple(
            float(value) + spatial_offset for value in original_slice_range
        )
    context_latent_slices = int(
        spatial_loss_cfg.get("context_latent_slices", 0)
    )
    if context_latent_slices < 0:
        raise ValueError("spatial_loss.context_latent_slices must be >= 0")
    if spatial_loss_enabled and context_latent_slices and adv_cfg.get("enabled", False):
        raise ValueError(
            "Evaluation-slab input cropping is currently a Stage-1-only "
            "experiment; disable train.adv"
        )

    output_dir = Path(cfg.get("output_dir", "runs/pretrain_t1w"))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        from utils.visualization import archive_metrics_csvs
        archive_metrics_csvs(output_dir / "vis")

    # ── Build bridge ── #
    resume_ckpt = args.resume
    ckpt_path = resume_ckpt or args.pretrain_ckpt or bridge_cfg.get("ckpt_path")

    num_train_timesteps = bridge_cfg.get("num_train_timesteps", 1000)

    # Note: scale_factor uses MAISI pretrained default (0.97045) from config.
    # We first build bridge with scale_factor=1.0, then update from config below.
    bridge = FieldRFModel.from_pretrained(
        ckpt_path,
        unet_channels=tuple(bridge_cfg.get("unet_channels", [64, 128, 256, 512])),
        num_res_blocks=tuple(bridge_cfg.get("num_res_blocks", [2, 2, 2, 2])),
        attention_levels=tuple(bridge_cfg.get("attention_levels", [False, False, True, True])),
        num_head_channels=tuple(bridge_cfg.get("num_head_channels", [0, 0, 32, 32])),
        norm_num_groups=bridge_cfg.get("norm_num_groups", 32),
        norm_eps=bridge_cfg.get("norm_eps", 1e-6),
        num_class_embeds=NUM_CLASSES,
        feat4_channels=bridge_cfg.get("feat4_channels", 64),
        conditioning_embedding_num_channels=tuple(
            bridge_cfg.get("controlnet", {}).get(
                "conditioning_embedding_num_channels", [64, 64]
            )
        ),
        z_channels=4,
        num_train_timesteps=num_train_timesteps,
        scale_factor=1.0,  # will be updated after data scan
        use_source_latent=bridge_cfg.get("use_source_latent", False),
        source_injection=bridge_cfg.get("source_injection"),
        source_adapter_channels=bridge_cfg.get("source_adapter_channels", 16),
        use_flash_attention=bridge_cfg.get("use_flash_attention", True),
        device=device,
        load_ema=bool(
            not resume_ckpt
            and bridge_cfg.get("load_pretrain_ema", True)
        ),
    )

    if args.dist:
        # find_unused_parameters=False: identity RF Stage 1 has no dead parameters
        # (unet + controlnet both used in forward). Setting True adds DDP graph
        # traversal overhead with no benefit. Re-enable only if source-conditioned RF's z_src
        # path introduces unused params.
        bridge = DDP(bridge, device_ids=[local_rank], find_unused_parameters=False)

    bridge_raw = bridge.module if args.dist else bridge

    # ── EMA ── #
    ema = None
    ema_decay = train_cfg.get("ema_decay", 0.999)
    if ema_decay > 0 and is_main:
        ema = EMA(bridge_raw, decay=ema_decay)

    # ── Data ── #
    cache_root = data_cfg["cache_root"]
    modalities = data_cfg.get("modalities", [data_cfg["modality"]])
    fields = data_cfg["fields"]
    feat_cache_name = data_cfg.get("feat_cache_name", "brainid_feat")
    train_split = data_cfg.get("split", "Training_retrospective")

    train_ds = MultiFieldCacheDataset(
        cache_root=cache_root,
        split=train_split,
        modalities=modalities,
        field_names=fields,
        feat_cache_name=feat_cache_name,
        max_volumes=args.max_volumes,
        graft=adv_cfg.get("enabled", False),
        cross_field=adv_cfg.get("cross_field", True),
        cross_modality=adv_cfg.get("cross_modality", True),
    )

    sampler = DistributedSampler(train_ds, shuffle=True) if args.dist else None
    num_workers = train_cfg.get("num_workers", 4)
    loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
        persistent_workers=(num_workers > 0),
    )

    if is_main:
        print(f"Data: {len(train_ds)} samples | modalities={modalities} | fields={fields}")
        print(f"Feats: 5 levels (feat[0-4]) from {feat_cache_name}")
        print(f"Source latent injection: {bridge_raw.source_injection}")
        if spatial_loss_enabled:
            print(
                "RF spatial supervision: original axial "
                f"[{original_slice_range[0]}, {original_slice_range[1]}) -> "
                "preprocessed axial "
                f"[{preprocessed_slice_range[0]}, {preprocessed_slice_range[1]}) "
                f"within {preprocessed_axis_size}",
                flush=True,
            )

    # ── scale_factor ── #
    # Use MAISI pretrained default directly (verified consistent with our data).
    # Avoids slow per-run computation (was loading 100 b2nd files = ~10min I/O).
    # If "auto" is set, fall back to data computation (all modalities, not just T1).
    scale_factor_cfg = vae_cfg.get("scale_factor", 1.0055984258651733)
    if scale_factor_cfg == "auto":
        if is_main:
            print("Computing scale_factor from data (100 random samples, ALL modalities)...", flush=True)
        scale_factor_candidates = []
        sample_count = 0
        temp_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=0,
        )
        with torch.no_grad():
            for batch in temp_loader:
                z = batch["z"]  # (1, 4, 96, 96, 96)
                scale_factor_candidates.append(1.0 / torch.std(z).item())
                sample_count += 1
                if sample_count >= 100:
                    break
        scale_factor = float(np.median(scale_factor_candidates))
        if is_main:
            print(f"scale_factor = {scale_factor:.6f} (computed from {sample_count} samples, ALL modalities)")
    else:
        scale_factor = float(scale_factor_cfg)
        if is_main:
            print(f"scale_factor = {scale_factor:.6f} (from config, project-wide verified value)")

    # Update bridge's scale_factor
    bridge_raw.scale_factor = scale_factor

    if adv_cfg.get("enabled", False) and not bridge_raw.use_source_latent:
        raise ValueError(
            "source-conditioned RF Stage 2 requires bridge.use_source_latent=true; "
            "otherwise graft z_src would be silently ignored."
        )
    source_dropout = float(adv_cfg.get("source_dropout", 0.1))
    if not 0.0 <= source_dropout <= 1.0:
        raise ValueError(f"source_dropout must be in [0,1], got {source_dropout}")

    # ── Discriminator (Stage 2, unused in identity RF) ── #
    disc = None
    opt_D = None
    if adv_cfg.get("enabled", False):
        disc_cfg = cfg.get("discriminator", {})
        disc = LatentProjectionDiscriminator(
            in_channels=4,
            num_classes=disc_cfg.get("num_class_embeds", NUM_CLASSES),
            base_ch=disc_cfg.get("base_channels", 64),
            use_spectral_norm=disc_cfg.get("use_spectral_norm", False),
        ).to(device)
        if args.dist:
            disc = DDP(disc, device_ids=[local_rank])
        d_lr = adv_cfg.get("d_lr", train_cfg["lr"])
        opt_D = torch.optim.Adam(
            (disc.module if args.dist else disc).parameters(),
            lr=d_lr, betas=tuple(adv_cfg.get("d_betas", [0.5, 0.999]))
        )

    # ── Optimizer and LR scheduler ── #
    n_epochs = args.n_epochs or train_cfg.get("n_epochs", 100)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    lr_min = train_cfg.get("lr_min", 1e-6)
    lr_base = train_cfg["lr"]
    steps_per_epoch = len(loader)
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    freeze_backbone_steps = int(
        adv_cfg.get("freeze_backbone_epochs", 0) * steps_per_epoch
        if bridge_raw.source_injection == "adapter" else 0
    )

    adapter_params = []
    backbone_params = list(bridge_raw.parameters())
    if bridge_raw.source_injection == "adapter":
        adapter_params = list(bridge_raw.source_adapter.parameters())
        adapter_ids = {id(p) for p in adapter_params}
        backbone_params = [p for p in bridge_raw.parameters() if id(p) not in adapter_ids]
        backbone_lr = float(adv_cfg.get("backbone_lr", lr_base))
        adapter_lr = float(adv_cfg.get("source_adapter_lr", lr_base))
        opt_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": adapter_params, "lr": adapter_lr, "name": "source_adapter"},
        ]
        group_lrs = [backbone_lr, adapter_lr]
    else:
        opt_groups = [{"params": backbone_params, "lr": lr_base, "name": "bridge"}]
        group_lrs = [lr_base]

    opt_G = torch.optim.AdamW(
        opt_groups,
        weight_decay=train_cfg.get("weight_decay", 0),
    )

    def common_lr_factor(step, group_lr):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        floor = min(lr_min / group_lr, 1.0)
        return floor + (1 - floor) * cos

    lr_lambdas = []
    for group_index, group_lr in enumerate(group_lrs):
        is_frozen_backbone = (
            bridge_raw.source_injection == "adapter" and group_index == 0
        )

        def group_lambda(step, group_lr=group_lr,
                         is_frozen_backbone=is_frozen_backbone):
            if is_frozen_backbone and step < freeze_backbone_steps:
                return 0.0
            return common_lr_factor(step, group_lr)

        lr_lambdas.append(group_lambda)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambdas)

    if is_main and bridge_raw.source_injection == "adapter":
        print(
            "Stage2 adapter schedule: "
            f"backbone_lr={group_lrs[0]:.2e} adapter_lr={group_lrs[1]:.2e} "
            f"freeze_backbone_epochs={adv_cfg.get('freeze_backbone_epochs', 0)} "
            f"graft_steps={adv_cfg.get('n_infer_steps', 4)} "
            f"adv_warmup_epochs={adv_cfg.get('warmup_epochs', 0)}",
            flush=True,
        )

    # ── Resume from checkpoint ── #
    start_epoch = 0
    if resume_ckpt:
        sd = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        bridge_raw.load_state_dict(sd["bridge"])
        opt_G.load_state_dict(sd["optimizer"])
        scheduler_G.load_state_dict(sd["scheduler"])
        if ema is not None and "ema" in sd:
            ema.load_state_dict(sd["ema"])
        if disc is not None and "disc" in sd:
            (disc.module if args.dist else disc).load_state_dict(sd["disc"])
            opt_D.load_state_dict(sd["opt_D"])
        start_epoch = sd["epoch"] + 1
        if "scale_factor" in sd:
            scale_factor = sd["scale_factor"]
            bridge_raw.scale_factor = scale_factor
            print(f"[RF] Restored scale_factor={scale_factor:.6f} from checkpoint")
        if is_main:
            print(f"[RF] Resumed from {resume_ckpt} -> epoch {start_epoch}", flush=True)

    # ── Training loop ── #
    t_start = time.time()

    for epoch in range(start_epoch, n_epochs):
        bridge_raw.train()
        if disc is not None:
            (disc.module if args.dist else disc).train()
        if args.dist:
            sampler.set_epoch(epoch)

        t0 = time.time()
        ep_loss = 0.0
        ep_recon = 0.0
        ep_g_adv = 0.0
        ep_identity = 0.0
        ep_d = 0.0
        ep_d_real = 0.0
        ep_d_fake = 0.0
        ep_d_fake_margin = 0.0
        ep_d_real_margin = 0.0
        ep_mean_gap = 0.0
        ep_std_gap = 0.0
        ep_source_drop = 0.0
        ep_adapter_grad = 0.0
        ep_backbone_grad = 0.0
        ep_lambda_adv = 0.0
        n_grad_diagnostics = 0
        n_steps = 0

        for step, batch in enumerate(loader):
            z = batch["z"].to(device)          # (B, 4, 96³) unscaled
            # FieldRFModel conditions only on the last BrainID feature. Keeping
            # earlier levels on CPU avoids transferring unused 3D tensors.
            feats = [batch["feats"][-1].to(device)]
            class_label = batch["class_label"].to(device)

            if spatial_loss_enabled and spatial_axis_weights is None:
                full_spatial_weights = spatial_axis_overlap_weights(
                    z.shape[spatial_axis],
                    preprocessed_slice_range,
                    preprocessed_axis_size,
                    device=device,
                )
                spatial_axis_weights = full_spatial_weights
                if context_latent_slices:
                    if context_latent_slices > z.shape[spatial_axis]:
                        raise ValueError(
                            "spatial_loss.context_latent_slices exceeds the "
                            "latent axial size"
                        )
                    active_full = torch.nonzero(
                        full_spatial_weights > 0, as_tuple=False
                    ).flatten()
                    required = int(active_full[-1] - active_full[0] + 1)
                    if context_latent_slices < required:
                        raise ValueError(
                            "spatial_loss.context_latent_slices must cover all "
                            f"evaluation cells ({required})"
                        )
                    crop_start = max(
                        0,
                        int(active_full[0])
                        - (context_latent_slices - required) // 2,
                    )
                    crop_start = min(
                        crop_start,
                        z.shape[spatial_axis] - context_latent_slices,
                    )
                    spatial_crop_bounds = (
                        crop_start,
                        crop_start + context_latent_slices,
                    )
                    spatial_axis_weights = full_spatial_weights[
                        spatial_crop_bounds[0]:spatial_crop_bounds[1]
                    ]
                if is_main:
                    active = torch.nonzero(
                        spatial_axis_weights > 0, as_tuple=False
                    ).flatten()
                    print(
                        "RF latent supervision: axial cells "
                        f"{active[0].item()}..{active[-1].item()} "
                        f"weights={spatial_axis_weights[active].tolist()} "
                        f"crop={spatial_crop_bounds or 'full-volume'}",
                        flush=True,
                    )

            # Preserve the full 96-cubed MAISI timestep distribution, then
            # crop only the network input.  The 24-cell context slab surrounds
            # the exact 7.5-cell (=30 image slices) supervised core.
            sampled_t = None
            if spatial_crop_bounds is not None:
                sampled_t = bridge_raw.sample_train_timesteps(z)
                crop_start, crop_end = spatial_crop_bounds
                z = z[..., crop_start:crop_end].contiguous()
                feature_factor = bridge_raw.conditioning_downsample_factor
                feats[-1] = feats[-1][
                    ...,
                    crop_start * feature_factor:crop_end * feature_factor,
                ].contiguous()

            opt_G.zero_grad(set_to_none=True)

            # Backpropagate the three G terms sequentially. Their gradients
            # still add in the same optimizer step, but each activation graph
            # is released before building the next one. Keeping recon, a
            # multi-step graft, and identity graphs alive together exceeds an
            # A100-80G at the native 96^3 latent resolution.
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
                loss_recon = rectified_flow_loss(bridge_raw, z, feats, class_label,
                                        scale_factor=scale_factor, z_src=None,
                                        t=sampled_t,
                                        spatial_axis_weights=spatial_axis_weights,
                                        spatial_axis=spatial_axis)
            loss_recon_val = loss_recon.detach().item()
            loss_recon.backward()
            loss_total_val = loss_recon_val

            # ── Stage 2 / source-conditioned RF: source-latent graft ── #
            adv_g_val = 0.0
            id_g_val = 0.0
            loss_d_val = 0.0
            d_real_val = 0.0
            d_fake_val = 0.0
            d_fake_margin_val = 0.0
            d_real_margin_val = 0.0
            mean_gap_val = 0.0
            std_gap_val = 0.0
            source_drop_val = 0.0
            lambda_adv_eff = 0.0
            z_fake_scaled = None
            if adv_cfg.get("enabled", False) and disc is not None:
                graft_feats = [batch["graft_feats"][-1].to(device)]
                graft_class = batch["graft_class"].to(device)
                graft_z_tgt = batch["graft_z_tgt_real"].to(device) * scale_factor
                graft_z_src = batch["graft_z_src"].to(device)
                graft_class_src = batch["graft_class_src"].to(device)

                adv_warmup_steps = int(
                    adv_cfg.get("warmup_epochs", 0) * steps_per_epoch
                )
                adv_progress = (
                    1.0 if adv_warmup_steps <= 0 else
                    min(1.0, (epoch * steps_per_epoch + step + 1) /
                        adv_warmup_steps)
                )
                lambda_adv_eff = float(adv_cfg.get("lambda_adv", 0.1)) * adv_progress
                disc_raw = disc.module if args.dist else disc

                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
                    # Adapter mode uses a pure-noise differentiable Euler
                    # trajectory; concat mode retains legacy random-t behavior.
                    z_fake_scaled, graft_diag = bridge_raw.generate_train(
                        graft_feats,
                        graft_class,
                        z_src=graft_z_src,
                        source_dropout_p=source_dropout,
                        n_infer_steps=int(adv_cfg.get("n_infer_steps", 4)),
                        return_diagnostics=True,
                    )
                    # D is a fixed differentiable critic for the G update.
                    for p in disc_raw.parameters():
                        p.requires_grad_(False)
                    fake_patch, fake_proj, _ = disc_raw(z_fake_scaled, graft_class)
                    fake_logits_g = fake_patch + fake_proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    adv_g = lsgan_g_loss(fake_logits_g)
                    weighted_adv = lambda_adv_eff * adv_g

                source_drop_val = graft_diag["source_drop_fraction"].item()
                adv_g_val = adv_g.detach().item()
                loss_total_val += weighted_adv.detach().item()
                weighted_adv.backward()
                # Keep only values needed by D; release the graft graph before
                # constructing the full-resolution identity graph.
                z_fake_scaled = z_fake_scaled.detach()

                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
                    # Supervise the adapter in the same active mode used by
                    # validation. Legacy concat stays unchanged as a control.
                    identity_z_src = (
                        graft_z_src
                        if bridge_raw.source_injection == "adapter" else None
                    )
                    loss_id = rectified_flow_loss(bridge_raw, graft_z_src, graft_feats,
                                         graft_class_src, scale_factor=scale_factor,
                                         z_src=identity_z_src,
                                         spatial_axis_weights=spatial_axis_weights,
                                         spatial_axis=spatial_axis)
                    weighted_id = adv_cfg.get("lambda_identity", 0.2) * loss_id

                id_g_val = loss_id.detach().item()
                loss_total_val += weighted_id.detach().item()
                weighted_id.backward()

            log_every = train_cfg.get("log_every", 50)
            sample_grad_diagnostic = is_main and (step + 1) % log_every == 0
            adapter_grad_val = (
                _grad_norm(adapter_params) if sample_grad_diagnostic else 0.0
            )
            backbone_grad_val = (
                _grad_norm(backbone_params) if sample_grad_diagnostic else 0.0
            )

            # LR=0 alone still lets Adam accumulate moments and lets frozen
            # backbone gradients dominate global clipping. Discard them after
            # logging so the first two epochs are a true optimizer freeze.
            if (bridge_raw.source_injection == "adapter" and
                    opt_G.param_groups[0]["lr"] == 0.0):
                for parameter in backbone_params:
                    parameter.grad = None

            if z_fake_scaled is not None:
                disc_raw = disc.module if args.dist else disc
                for p in disc_raw.parameters():
                    p.requires_grad_(True)
                opt_D.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=amp_dtype):
                    z_fake_det = z_fake_scaled.detach()
                    fake_patch_d, fake_proj_d, _ = disc(z_fake_det, graft_class)
                    fake_logits_d = fake_patch_d + fake_proj_d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    real_patch_d, real_proj_d, _ = disc(graft_z_tgt, graft_class)
                    real_logits_d = real_patch_d + real_proj_d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    loss_d = lsgan_d_loss(real_logits_d, fake_logits_d)

                    mismatch_weight = float(adv_cfg.get("lambda_mismatch", 0.0))
                    real_wrong_patch, real_wrong_proj, _ = disc(
                        graft_z_tgt, graft_class_src
                    )
                    real_wrong_logits = (
                        real_wrong_patch +
                        real_wrong_proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    )
                    if mismatch_weight > 0:
                        loss_d = loss_d + mismatch_weight * 0.5 * F.mse_loss(
                            real_wrong_logits, torch.zeros_like(real_wrong_logits)
                        )

                    # Conditional-use diagnostic: the same fake should score
                    # higher for its requested target than for its source class.
                    with torch.no_grad():
                        fake_wrong_logits = disc_raw.full_logits(
                            z_fake_det, graft_class_src
                        )

                loss_d.backward()
                opt_D.step()
                loss_d_val = loss_d.detach().item()
                d_real_val = real_logits_d.detach().float().mean().item()
                d_fake_val = fake_logits_d.detach().float().mean().item()
                d_fake_margin_val = (
                    fake_logits_d.detach().float().mean() -
                    fake_wrong_logits.detach().float().mean()
                ).item()
                d_real_margin_val = (
                    real_logits_d.detach().float().mean() -
                    real_wrong_logits.detach().float().mean()
                ).item()
                mean_gap_val, std_gap_val = _latent_moment_gap(
                    z_fake_det, graft_z_tgt
                )

            torch.nn.utils.clip_grad_norm_(bridge_raw.parameters(), 1.0)
            opt_G.step()
            scheduler_G.step()

            if ema is not None:
                ema.update(bridge_raw)

            ep_loss += loss_total_val
            ep_recon += loss_recon_val
            ep_g_adv += adv_g_val
            ep_identity += id_g_val
            ep_d += loss_d_val
            ep_d_real += d_real_val
            ep_d_fake += d_fake_val
            ep_d_fake_margin += d_fake_margin_val
            ep_d_real_margin += d_real_margin_val
            ep_mean_gap += mean_gap_val
            ep_std_gap += std_gap_val
            ep_source_drop += source_drop_val
            ep_adapter_grad += adapter_grad_val
            ep_backbone_grad += backbone_grad_val
            ep_lambda_adv += lambda_adv_eff
            n_grad_diagnostics += int(sample_grad_diagnostic)
            n_steps += 1

            # ── Logging ── #
            if is_main and (step + 1) % log_every == 0:
                lr_text = "/".join(
                    f"{group['lr']:.2e}" for group in opt_G.param_groups
                )
                print(f"  [{epoch+1}/{n_epochs}] step {step+1}/{len(loader)} "
                      f"loss={loss_total_val:.5f} recon={loss_recon_val:.5f} "
                      f"g_adv={adv_g_val:.5f} id={id_g_val:.5f} d={loss_d_val:.5f} "
                      f"adv_w={lambda_adv_eff:.4f} "
                      f"Dscore[r={d_real_val:.3f},f={d_fake_val:.3f},"
                      f"margin_real={d_real_margin_val:.3f},"
                      f"margin_fake={d_fake_margin_val:.3f}] "
                      f"mom=[mean={mean_gap_val:.4f},std={std_gap_val:.4f}] "
                      f"src_drop={source_drop_val:.2f} "
                      f"grad=[adapter={adapter_grad_val:.3e},base={backbone_grad_val:.3e}] "
                      f"lr={lr_text} ({time.time()-t0:.0f}s)", flush=True)

        # ── Epoch end ── #

        if is_main:
            dt = time.time() - t0
            cum = time.time() - t_start
            lr_text = "/".join(
                f"{group['lr']:.2e}" for group in opt_G.param_groups
            )
            print(f"Epoch {epoch+1}/{n_epochs} loss={ep_loss/n_steps:.5f} "
                  f"recon={ep_recon/n_steps:.5f} "
                  f"g_adv={ep_g_adv/n_steps:.5f} "
                  f"id={ep_identity/n_steps:.5f} d={ep_d/n_steps:.5f} "
                  f"adv_w={ep_lambda_adv/n_steps:.4f} "
                  f"Dscore[r={ep_d_real/n_steps:.3f},f={ep_d_fake/n_steps:.3f},"
                  f"margin_real={ep_d_real_margin/n_steps:.3f},"
                  f"margin_fake={ep_d_fake_margin/n_steps:.3f}] "
                  f"mom=[mean={ep_mean_gap/n_steps:.4f},std={ep_std_gap/n_steps:.4f}] "
                  f"src_drop={ep_source_drop/n_steps:.3f} "
                  f"grad=[adapter={ep_adapter_grad/max(n_grad_diagnostics, 1):.3e},"
                  f"base={ep_backbone_grad/max(n_grad_diagnostics, 1):.3e}] "
                  f"lr={lr_text} "
                  f"time={dt:.0f}s cum={cum:.0f}s"
                  f"{source_channel_norms(bridge_raw)}"
                  f"{controlnet_norms(bridge_raw)}", flush=True)

            # ── Save checkpoint ── #
            save_every = train_cfg.get("save_every", 10)
            if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
                ckpt_path = output_dir / f"fieldrf_epoch{epoch+1}.pt"
                state = {
                    "bridge": bridge_raw.state_dict(),
                    "optimizer": opt_G.state_dict(),
                    "scheduler": scheduler_G.state_dict(),
                    "epoch": epoch,
                    "config": cfg,
                    "scale_factor": scale_factor,
                    "num_train_timesteps": num_train_timesteps,
                    "bridge_config": {
                        "unet_channels": tuple(bridge_cfg.get("unet_channels", [64, 128, 256, 512])),
                        "num_res_blocks": tuple(bridge_cfg.get("num_res_blocks", [2, 2, 2, 2])),
                        "attention_levels": tuple(bridge_cfg.get("attention_levels", [False, False, True, True])),
                        "num_head_channels": tuple(bridge_cfg.get("num_head_channels", [0, 0, 32, 32])),
                        "norm_num_groups": bridge_cfg.get("norm_num_groups", 32),
                        "feat4_channels": bridge_cfg.get("feat4_channels", 64),
                        "conditioning_embedding_num_channels": tuple(
                            bridge_cfg.get("controlnet", {}).get(
                                "conditioning_embedding_num_channels", [64, 64]
                            )
                        ),
                        "use_flash_attention": bridge_cfg.get(
                            "use_flash_attention", True
                        ),
                        "num_train_timesteps": num_train_timesteps,
                        "use_source_latent": bridge_raw.use_source_latent,
                        "source_injection": bridge_raw.source_injection,
                        "source_adapter_channels": bridge_cfg.get(
                            "source_adapter_channels", 16
                        ),
                    },
                }
                if ema is not None:
                    state["ema"] = ema.state_dict()
                if disc is not None:
                    state["disc"] = (disc.module if args.dist else disc).state_dict()
                    state["opt_D"] = opt_D.state_dict()
                torch.save(state, ckpt_path)
                print(f"  Saved: {ckpt_path}", flush=True)

            # ── Visualization ── #
            vis_every = vis_cfg.get("vis_every", 10)
            if (epoch + 1) % vis_every == 0:
                from visualize import run_vis
                if ema is not None:
                    backup = ema.apply_to(bridge_raw)
                    try:
                        run_vis(bridge_raw, scale_factor, cfg, epoch, device)
                    finally:
                        ema.restore(bridge_raw, backup)
                else:
                    run_vis(bridge_raw, scale_factor, cfg, epoch, device)

    if args.dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
