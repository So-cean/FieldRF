"""Finetune MAISI VAE decoder on MRIxFields2026 using cached latents.

Key insight (verified by analysis/verify_vae_encoder.py):
  Encoder produces identical-quality latents for T1W/T2W/T2FLAIR.
  The reconstruction SSIM gap is a DECODER issue.
  Therefore we freeze the encoder and only finetune the decoder.

All paths read from YAML config, no hardcoded constants.

Usage:
    python scripts/finetune_vae.py --config configs/vae_finetune.yaml
    python scripts/finetune_vae.py --config configs/vae_finetune.yaml --max_volumes 5 --epochs 1  # smoke
"""

import argparse
import copy
import os
import sys
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import yaml

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.visualization import (
    load_vae, load_orig_ras_384, decode_volume_centered,
    eval_window_in_384, find_nifti_path, compute_metrics_eval,
    init_metrics_csv, render_vis_grid_vae, archive_metrics_csvs,
)
from models.checkpoint import load_vae_checkpoint


# ── Dataset ── #

class LatentImageDataset(torch.utils.data.Dataset):
    """Loads cached VAE latent + original NIfTI, returns matching patches.

    Cached latent: (4, 96, 96, 96) from .pt file.
    Original NIfTI: 384^3, [0,1] after RAS + pad/crop.
    Patch: latent (patch_size//4)^3 -> image patch_size^3, same spatial location.
    """

    def __init__(self, cache_root, data_root, split, modalities, fields,
                 patch_size=128, max_volumes=0):
        self.patch_size = patch_size
        self.latent_patch = patch_size // 4
        self.files = []
        # max_volumes is PER (modality, field) combo (0 = no cap)
        per_mod_field = max_volumes if max_volumes > 0 else 0
        for mod in modalities:
            for field in fields:
                latent_dir = Path(cache_root) / split / "latent" / mod / field
                if not latent_dir.is_dir():
                    continue
                pts = sorted(latent_dir.glob("*.pt"))
                if per_mod_field > 0:
                    pts = pts[:per_mod_field]
                for pt in pts:
                    stem = pt.stem
                    nii = Path(data_root) / split / mod / field / f"{stem}.nii.gz"
                    if nii.exists():
                        self.files.append({"latent": str(pt), "nii": str(nii),
                                           "modality": mod, "field": field})
        mod_counts = {}
        for f in self.files:
            mod_counts[f["modality"]] = mod_counts.get(f["modality"], 0) + 1
        for mod, count in sorted(mod_counts.items()):
            print(f"  {mod}: {count} volumes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        d = torch.load(item["latent"], map_location="cpu", weights_only=False)
        z = d["latent"]  # (4, 96, 96, 96)

        orig_384, _, _, _ = load_orig_ras_384(item["nii"])  # (384,384,384) [0,1]

        # Random crop (latent coords -> image coords via *4)
        D, H, W = z.shape[1:]
        pd = self.latent_patch
        dz = random.randint(0, D - pd)
        dy = random.randint(0, H - pd)
        dx = random.randint(0, W - pd)

        z_patch = z[:, dz:dz+pd, dy:dy+pd, dx:dx+pd]
        ip = self.patch_size
        img_patch = orig_384[dz*4:dz*4+ip, dy*4:dy*4+ip, dx*4:dx*4+ip]

        # Augmentation: random flip (both together)
        for axis in range(3):
            if random.random() < 0.5:
                z_patch = torch.flip(z_patch, [axis + 1])
                img_patch = np.flip(img_patch, axis=axis).copy()

        img_patch = img_patch * 2 - 1  # [0,1] -> [-1,1]
        return {
            "latent": z_patch,
            "image": torch.from_numpy(img_patch.copy()).float().unsqueeze(0),
            "modality": item["modality"],
            "field": item["field"],
        }


# ── VAE ── #

def build_vae(vae_cfg):
    """Build MAISI VAE from config (params match load_vae in visualization.py)."""
    return AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
        num_channels=vae_cfg.get("num_channels", [64, 128, 256]),
        num_res_blocks=vae_cfg.get("num_res_blocks", [2, 2, 2]),
        norm_num_groups=vae_cfg.get("norm_num_groups", 32),
        norm_eps=vae_cfg.get("norm_eps", 1e-6),
        attention_levels=vae_cfg.get("attention_levels", [False, False, False]),
        with_encoder_nonlocal_attn=vae_cfg.get("with_encoder_nonlocal_attn", False),
        with_decoder_nonlocal_attn=vae_cfg.get("with_decoder_nonlocal_attn", False),
        use_checkpointing=vae_cfg.get("use_checkpointing", False),
        use_convtranspose=vae_cfg.get("use_convtranspose", False),
        norm_float16=vae_cfg.get("norm_float16", True),
        num_splits=vae_cfg.get("num_splits", 4),
        dim_split=vae_cfg.get("dim_split", 1),
    )


def load_pretrained(vae, ckpt_path):
    load_vae_checkpoint(vae, ckpt_path)
    print("[VAE] Loaded complete pretrained state (strict architecture check)")


# ── Validation (RF-aligned, 2-column VAE grid) ── #

@torch.no_grad()
def validate_vae(vae, cfg, epoch, device, output_dir):
    """RF-aligned validation: decode full latent, compare with 30-slice eval window.

    Iterates over ALL fields (from data.fields) × modalities, 1 subject each.
    Uses find_nifti_path, eval_window_in_384, render_vis_grid_vae (2-column).

    Outputs per epoch (per modality):
      vae_val_epoch{N}_{mod}.png   — 2-column grid (Orig | Decoded)
      vae_val_metrics_{mod}.csv    — epoch, field, subject, nrmse_vae, ssim_vae, psnr_vae
    """
    vis_cfg = cfg.get("vis", {})
    data_cfg = cfg["data"]
    vae_cfg = cfg["vae"]

    eval_range = tuple(vis_cfg.get("eval_slice_range", [150, 180]))
    vis_split = vis_cfg.get("split", "Validating_prospective")
    cache_root = data_cfg["cache_root"]
    fields = data_cfg.get("fields", ["0.1T", "1.5T", "3T", "5T", "7T"])
    modalities = data_cfg.get("modalities", ["T1W"])
    n_samples = vis_cfg.get("n_samples", 1)  # subjects per (modality, field)

    sw_cfg = vae_cfg.get("sliding_window", {"roi_size": [128, 128, 64]})
    amp_dtype = getattr(torch, vae_cfg.get("sliding_window", {}).get("amp_dtype", "float16"))

    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    vae.eval()

    mod_rows = {}  # mod -> list of rows_data entries (for PNG grid: first subject only)
    all_ssim_list = []
    all_n = 0

    for mod in modalities:
        csv_path = vis_dir / f"vae_val_metrics_{mod}.csv"
        writer, csv_f = init_metrics_csv(
            csv_path,
            ["epoch", "field", "subject", "nrmse_vae", "ssim_vae", "psnr_vae"],
        )

        rows_data = []  # first subject per field (for PNG)
        for field in fields:
            latent_dir = Path(cache_root) / vis_split / "latent" / mod / field
            if not latent_dir.is_dir():
                continue
            pt_files = sorted(latent_dir.glob("*.pt"))
            if not pt_files:
                continue
            pt_files = pt_files[:n_samples]

            field_ssims = []
            for pi, pt in enumerate(pt_files):
                subj_id = pt.stem.rsplit("_", 1)[-1]
                nii_path = find_nifti_path(data_cfg["data_root"], vis_split, mod, field, subj_id)
                if nii_path is None:
                    continue

                z = torch.load(str(pt), map_location="cpu", weights_only=False)["latent"]
                orig_384, _, pad_before_z, crop_start_z = load_orig_ras_384(nii_path)

                z0, z1 = eval_window_in_384(pad_before_z, crop_start_z, eval_range)
                z_center = (z0 + z1) // 2

                recon_img = decode_volume_centered(
                    vae, z.unsqueeze(0).to(device), sw_cfg, amp_dtype, z_center
                )
                recon_np = torch.clamp((recon_img + 1) / 2, 0, 1)[0, 0].float().cpu().numpy()

                win_start = round((z_center - 32) / 4) * 4
                c0 = max(0, z0 - win_start)
                c1 = min(recon_np.shape[2], z1 - win_start)
                recon_slab = recon_np[:, :, c0:c1]
                orig_slab = orig_384[:, :, z0:z1]

                brain_mask = (orig_slab > 0).astype(np.float32)
                recon_m = recon_slab * brain_mask
                orig_m = orig_slab * brain_mask

                m = compute_metrics_eval(recon_m, orig_m, slice_axis=2)

                writer.writerow([epoch + 1, field, subj_id,
                                 f"{m['nrmse']:.4f}", f"{m['ssim']:.4f}", f"{m['psnr']:.2f}"])

                field_ssims.append(m['ssim'])
                all_ssim_list.append(m['ssim'])
                all_n += 1

                # First subject only → PNG grid
                if pi == 0:
                    rows_data.append({
                        "orig": orig_m, "decoded": recon_m,
                        "field": field, "subject": subj_id,
                        "metrics": m,
                    })

                print(f"    [{mod}@{field}/{subj_id}] ssim={m['ssim']:.4f} "
                      f"psnr={m['psnr']:.2f} nrmse={m['nrmse']:.4f}", flush=True)

            if field_ssims:
                avg_f = float(np.mean(field_ssims))
                print(f"    → {mod}@{field} avg_ssim={avg_f:.4f} ({len(field_ssims)} subjects)", flush=True)

        mod_rows[mod] = rows_data
        csv_f.close()

        if rows_data:
            vis_path = vis_dir / f"vae_val_epoch{epoch+1:03d}_{mod}.png"
            render_vis_grid_vae(
                rows_data,
                f"VAE Decoder Validation — {mod} — Epoch {epoch+1}",
                vis_path, epoch,
            )

    vae.train()

    if all_n > 0:
        avg_all = float(np.mean(all_ssim_list))
        print(f"  ALL: avg_ssim={avg_all:.4f} ({all_n} samples)", flush=True)
        return {"ssim": avg_all, "n": all_n}
    return {"ssim": 0, "n": 0}


# ── Distributed ── #

def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr); dist.init_process_group(backend="nccl")
        return rank, ws, lr
    return 0, 1, 0


# ── Main ── #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to VAE finetune YAML config")
    p.add_argument("--max_volumes", type=int, default=0,
                   help="Max volumes per (modality, field) for smoke test (0 = all)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs from config (for smoke test)")
    p.add_argument("--resume_ckpt", default=None)
    p.add_argument("--dist", action="store_true")
    args = p.parse_args()

    # ── Load config ──
    fieldrf_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = fieldrf_root / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    vae_cfg = cfg["vae"]
    train_cfg = cfg["train"]
    vis_cfg = cfg.get("vis", {})

    # CLI overrides for smoke test
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs

    rank, world_size, local_rank = setup_distributed() if args.dist else (0, 1, 0)
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.get("output_dir", "runs/vae_finetune"))
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Archive old metrics CSVs (prevents mixing runs)
        archive_metrics_csvs(out_dir / "vis")
        # Archive old train metrics too (same CSV accumulation risk)
        archive_metrics_csvs(out_dir / "train_vis")

    # ── Data ──
    train_ds = LatentImageDataset(
        cache_root=data_cfg["cache_root"],
        data_root=data_cfg["data_root"],
        split=data_cfg["split"],
        modalities=data_cfg.get("modalities", ["T1W"]),
        fields=data_cfg.get("fields", ["0.1T", "1.5T", "3T", "5T", "7T"]),
        patch_size=train_cfg.get("patch_size", 128),
        max_volumes=args.max_volumes,
    )
    if len(train_ds) == 0:
        raise SystemExit("No cached latents found")

    sampler = DistributedSampler(train_ds, shuffle=True) if args.dist else None
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_cfg.get("batch_size", 2),
        shuffle=(sampler is None), sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        persistent_workers=train_cfg.get("num_workers", 4) > 0,
        pin_memory=True, drop_last=True,
    )

    if is_main:
        print(f"Data: {len(train_ds)} volumes | patch={train_cfg.get('patch_size', 128)}^3")
        print(f"Steps/epoch: {len(loader)} | batch={train_cfg.get('batch_size', 2)}")

    # ── VAE ──
    vae = build_vae(vae_cfg)
    init_ckpt = args.resume_ckpt or vae_cfg["checkpoint"]
    load_pretrained(vae, init_ckpt)

    # ── Freeze encoder ──
    if train_cfg.get("freeze_encoder", True):
        for param in vae.encoder.parameters():
            param.requires_grad_(False)
        n_frozen = sum(p.numel() for p in vae.encoder.parameters())
        n_train = sum(p.numel() for p in vae.decoder.parameters() if p.requires_grad)
        if is_main:
            print(f"[VAE] Encoder FROZEN ({n_frozen/1e6:.1f}M), decoder trainable ({n_train/1e6:.1f}M)")

    vae = vae.to(device)
    if args.dist:
        vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=not train_cfg.get("freeze_encoder", True))

    # ── Losses ──
    l1 = nn.L1Loss(reduction='none')  # per-sample for EMA loss balancing
    ssim_weight = train_cfg.get("ssim_weight", 0.5)
    perc_weight = train_cfg.get("perceptual_weight", 0.3)

    ssim_loss_fn = None
    if ssim_weight > 0:
        from monai.losses import SSIMLoss
        ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=11,
                                reduction='none')  # returns (B,1) per-sample
        if is_main: print(f"SSIM loss: weight={ssim_weight} (reduction='none')")

    perc = None
    if perc_weight > 0:
        try:
            import warnings
            with warnings.catch_warnings():
                # MONAI PerceptualLoss internally uses deprecated pretrained=True
                warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
                from monai.losses import PerceptualLoss
                perc = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                       is_fake_3d=True, fake_3d_ratio=0.2).to(device)
            perc.eval()
            if is_main: print(f"Perceptual loss: weight={perc_weight}")
        except Exception as e:
            raise RuntimeError(
                "Configured VAE perceptual loss could not be initialized. "
                "Check torchvision/lpips dependencies and pretrained SqueezeNet weights, "
                "or explicitly set train.perceptual_weight: 0 to disable this loss."
            ) from e

    # ── Optimizer (decoder only) ──
    trainable_params = [p for p in vae.parameters() if p.requires_grad]
    opt_G = torch.optim.Adam(trainable_params, lr=train_cfg["lr"], eps=1e-6)

    # ── Step-level scheduler ──
    # IMPORTANT: set_lr() must be called BEFORE the first opt_G.step() so warmup
    # is effective. Call once after creation (global_step=0 → lr=0 for step 0).
    epochs = train_cfg.get("epochs", 30)
    warmup_steps = train_cfg.get("warmup_steps", 0)
    min_lr = train_cfg.get("min_lr", 1e-7)
    base_lr = train_cfg["lr"]
    accum_steps = train_cfg.get("accum_steps", 4)
    global_step = 0  # tracks optimizer steps (after grad accum)

    def set_lr():
        """Set LR based on global step progress (step-level cosine decay)."""
        total_batches = len(loader) * epochs
        total_opt_steps = math.ceil(total_batches / accum_steps)
        if total_opt_steps == 0: return
        if global_step < warmup_steps:
            # Linear warmup: 0 → 1 over warmup_steps
            factor = global_step / max(warmup_steps, 1)
        else:
            p = min((global_step - warmup_steps) / max(total_opt_steps - warmup_steps, 1), 1.0)
            factor = min_lr / base_lr + (1 - min_lr / base_lr) * (1 + math.cos(math.pi * p)) / 2
        for pg in opt_G.param_groups:
            pg['lr'] = base_lr * factor
    set_lr()  # global_step=0 → lr=0, ensures first optimizer.step() is a no-op (warmup)
    save_every = train_cfg.get("save_every", 5)
    vis_every = vis_cfg.get("vis_every", 1)  # MUST be 1


    # ── EMA loss balancing (GradNorm-style) ──
    # Tracks per-(modality,field) EMA loss, auto-scales per-sample gradients so
    # high-loss classes (e.g. T2W 5T) don't dominate and wash out low-loss ones
    # (e.g. T1W 5T near-ceiling). Self-adaptive, no manual weight tuning.
    from collections import defaultdict
    balance_cfg = train_cfg.get("loss_balance", {})
    balance_enabled = balance_cfg.get("enabled", False)
    balance_decay = balance_cfg.get("decay", 0.99)
    balance_warmup = balance_cfg.get("warmup_steps", 100)
    balance_clip_min = balance_cfg.get("clip_min", 0.2)
    balance_clip_max = balance_cfg.get("clip_max", 5.0)
    ema_loss = defaultdict(lambda: 0.1)  # key = "T1W_5T"; realistic init, not 1.0
    balance_step = 0

    if is_main:
        bal_str = (f"enabled={balance_enabled}, decay={balance_decay}, "
                   f"warmup={balance_warmup}, clip=[{balance_clip_min},{balance_clip_max}]"
                   if balance_enabled else "disabled")
        print(f"Loss balance: {bal_str}")
        print(f"Training: {epochs} ep, lr={base_lr}->{min_lr}, "
              f"ssim={ssim_weight}, perc={perc_weight}")
        print(f"Warmup: {warmup_steps} steps | Accum: {accum_steps}")
        print(f"Validation: vis_every={vis_every} (MUST be 1)")

    # ── Epoch 0 validation (sanity check before training) ──
    if is_main and vis_every > 0:
        print("  --- Validation (epoch 0, before training) ---", flush=True)
        vae_sd = vae.module if args.dist else vae
        validate_vae(vae_sd, cfg, -1, device, out_dir)  # epoch=-1 → prints as "epoch 0"

    # ── Training loop ──
    t_start = time.time()
    for epoch in range(epochs):
        vae.train()
        if args.dist: sampler.set_epoch(epoch)
        t0 = time.time()
        ep_l1 = ep_ssim = ep_perc = 0.0
        n_steps = 0

        opt_G.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            z_patch = batch["latent"].to(device)
            x_patch = batch["image"].to(device)
            modalities = batch["modality"]  # list of strings (default_collate keeps strings)
            fields = batch["field"]
            batch_keys = list(modalities)  # group by modality only (3 groups: T1W, T2W, T2FLAIR)
            B = z_patch.shape[0]

            loss_perc_val = torch.tensor(0.0, device=device)

            # Full float32 forward (decoder is only 12M params, bf16 not needed)
            vae_raw = vae.module if args.dist else vae
            recon = vae_raw.decode(z_patch.float())  # (B,1,D,H,W)

            # ── Per-sample losses (reduction='none') ──
            loss_l1_per = l1(recon, x_patch).mean(dim=[1, 2, 3, 4])  # (B,)

            loss_ssim_per = torch.zeros(B, device=device)
            if ssim_loss_fn is not None:
                recon_01 = torch.clamp((recon + 1) / 2, 0, 1)
                x_01 = torch.clamp((x_patch + 1) / 2, 0, 1)
                loss_ssim_per = ssim_loss_fn(recon_01, x_01).flatten()  # (B,)

            # Perceptual: batch-level scalar (PerceptualLoss has no reduction='none')
            # Kept as graph tensor — added AFTER per-sample weighting so gradient flows
            loss_perc_val = torch.tensor(0.0, device=device)
            if perc is not None:
                loss_perc_val = perc(recon, x_patch)  # scalar, IN graph

            # Per-sample loss for balancing (L1+SSIM only — perceptual is batch-level)
            loss_per = loss_l1_per + ssim_weight * loss_ssim_per  # (B,)

            # ── EMA-based gradient balancing (GradNorm-style) ──
            balance_step += 1
            with torch.no_grad():
                for i, key in enumerate(batch_keys):
                    ema_loss[key] = (balance_decay * ema_loss[key]
                                     + (1 - balance_decay) * loss_per[i].item())

            if balance_enabled and balance_step > balance_warmup:
                # weight = mean_ema / ema_loss[class_i] → high-loss class gets low weight
                # Use GLOBAL mean of all classes (more stable than batch-only 2 samples)
                mean_ema = float(np.mean(list(ema_loss.values())))
                weights = torch.tensor(
                    [mean_ema / max(ema_loss[k], 1e-8) for k in batch_keys],
                    device=device, dtype=torch.float32)
                weights = weights.clamp(balance_clip_min, balance_clip_max)
            else:
                weights = torch.ones(B, device=device)

            # Weighted per-sample loss (L1+SSIM) + unweighted batch-level perceptual
            loss_g = (loss_per * weights).mean()
            if perc is not None:
                loss_g = loss_g + perc_weight * loss_perc_val  # gradient flows!

            (loss_g / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
                opt_G.step()
                global_step += 1
                set_lr()
                opt_G.zero_grad(set_to_none=True)

            ep_l1 += loss_l1_per.mean().item()
            ep_ssim += loss_ssim_per.mean().item()
            ep_perc += loss_perc_val.item()
            n_steps += 1

            if is_main and (step + 1) % 100 == 0:
                # Show per-class EMA + weights for debugging
                cls_str = " ".join(f"{k}:{ema_loss[k]:.4f}/w={w:.2f}"
                                   for k, w in zip(batch_keys, weights.tolist()))
                print(f"  [{epoch+1}/{epochs}] step {step+1}/{len(loader)} "
                      f"l1={loss_l1_per.mean().item():.5f} "
                      f"ssim={loss_ssim_per.mean().item():.5f} "
                      f"perc={loss_perc_val.item():.5f} ({time.time()-t0:.0f}s) "
                      f"[{cls_str}]", flush=True)

        # Flush residual gradients at epoch end (prevents last-batch gradient loss
        # when len(loader) % accum_steps != 0)
        if n_steps > 0 and n_steps % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            opt_G.step()
            global_step += 1
            set_lr()
            opt_G.zero_grad(set_to_none=True)

        if is_main:
            current_lr = opt_G.param_groups[0]['lr']
            dt = time.time() - t0
            cum = time.time() - t_start
            print(f"Epoch {epoch+1}/{epochs} "
                  f"l1={ep_l1/max(n_steps,1):.5f} ssim={ep_ssim/max(n_steps,1):.5f} "
                  f"perc={ep_perc/max(n_steps,1):.5f} "
                  f"lr={current_lr:.3e} "
                  f"time={dt:.0f}s cum={cum:.0f}s", flush=True)

            # Per-class EMA loss summary (verify balance is working)
            if balance_enabled and len(ema_loss) > 0:
                sorted_items = sorted(ema_loss.items(), key=lambda x: x[1], reverse=True)
                cls_summary = "  ".join(f"{k}={v:.4f}" for k, v in sorted_items)
                print(f"  EMA loss per class: {cls_summary}", flush=True)

            # Save checkpoint
            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                vae_sd = vae.module if args.dist else vae
                ckpt_path = out_dir / f"vae_finetune_epoch{epoch+1}.pt"
                torch.save(
                    {"state_dict": {f"autoencoder.{k}": v for k, v in vae_sd.state_dict().items()}},
                    ckpt_path,
                )
                print(f"  Saved: {ckpt_path}", flush=True)

            # Validation (every epoch, MUST be 1)
            if (epoch + 1) % vis_every == 0:
                print(f"  --- Validation (epoch {epoch+1}) ---", flush=True)
                vae_sd = vae.module if args.dist else vae
                val_metrics = validate_vae(vae_sd, cfg, epoch, device, out_dir)
                print(f"  Val epoch {epoch+1}: ssim={val_metrics['ssim']:.4f} "
                      f"({val_metrics['n']} samples)", flush=True)



    if is_main:
        vae_sd = vae.module if args.dist else vae
        torch.save(
            {"state_dict": {f"autoencoder.{k}": v for k, v in vae_sd.state_dict().items()}},
            out_dir / "autoencoder_finetuned.pt",
        )
        print(f"Done. Drop-in: {out_dir}/autoencoder_finetuned.pt")

    if args.dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
