"""Adapt the MAISI decode path to cached MRI latents.

The encoding path is frozen; checkpoints retain the full autoencoder weights
and persist optimizer, scheduler and per-rank state for epoch-boundary resume.
"""

import time
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import yaml

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from fieldrf.utils.visualization import archive_metrics_csvs
from fieldrf.visualization.vae import validate_vae
from fieldrf.models.checkpoint import load_vae_checkpoint


from fieldrf.data.vae_dataset import LatentImageDataset
from fieldrf.models.vae import prepare_decoder
from fieldrf.training.distributed import setup_distributed, wrap_module, synchronize
from fieldrf.training.checkpoints import (atomic_save, collect_rank_states, restore_rank_state, validate_vae_resume)
from fieldrf.training.schedules import WarmupCosineSchedule


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


# ── Main ── #

def run(args):

    # ── Load config ──
    fieldrf_root = Path(__file__).resolve().parents[2]
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

    rank, world_size, local_rank, device = setup_distributed(args.dist, args.device)
    is_main = rank == 0
    seed = int(train_cfg.get("seed", 2026))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    loader_generator = torch.Generator().manual_seed(seed + rank)
    out_dir = Path(cfg.get("output_dir", "runs/vae_finetune"))
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_main and not args.resume_ckpt:
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

    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed) if args.dist else None
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=train_cfg.get("batch_size", 2),
        shuffle=(sampler is None), sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        persistent_workers=False,
        generator=loader_generator,
        pin_memory=True, drop_last=True,
    )

    if len(loader) == 0:
        raise ValueError("VAE training has no batches; reduce batch_size or provide more volumes")

    if is_main:
        print(f"Data: {len(train_ds)} volumes | patch={train_cfg.get('patch_size', 128)}^3")
        print(f"Steps/epoch: {len(loader)} | batch={train_cfg.get('batch_size', 2)}")

    # ── VAE ──
    vae = build_vae(vae_cfg)
    init_ckpt = args.resume_ckpt or args.init_ckpt or vae_cfg["checkpoint"]
    resume_state = None
    if args.resume_ckpt:
        resume_state = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        validate_vae_resume(resume_state, train_cfg, len(loader))
    load_pretrained(vae, init_ckpt)

    if not train_cfg.get("freeze_encoder", True):
        raise ValueError("Cached-latent adaptation requires train.freeze_encoder: true")
    vae = vae.to(device)
    decoder = prepare_decoder(vae)
    if is_main:
        n_frozen = sum(p.numel() for p in vae.parameters() if not p.requires_grad)
        n_train = sum(p.numel() for p in decoder.parameters())
        print(f"[VAE] Encoding path FROZEN ({n_frozen/1e6:.1f}M), decode path trainable ({n_train/1e6:.1f}M)")
    if args.dist:
        decoder = wrap_module(decoder, device)

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
    # Initialize LR before the first optimizer update, preserving VAE warmup.
    epochs = train_cfg.get("epochs", 30)
    warmup_steps = train_cfg.get("warmup_steps", 0)
    min_lr = train_cfg.get("min_lr", 1e-7)
    base_lr = train_cfg["lr"]
    accum_steps = train_cfg.get("accum_steps", 4)
    global_step = 0  # tracks optimizer steps (after grad accum)

    scheduler = WarmupCosineSchedule(
        opt_G, epochs * math.ceil(len(loader) / accum_steps), warmup_steps, min_lr,
    )
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

    start_epoch = 0
    if resume_state is not None:
        opt_G.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state["global_step"])
        if global_step != scheduler.global_step:
            raise ValueError("VAE checkpoint optimizer/scheduler step counters differ")
        start_epoch = int(resume_state["epoch"]) + 1
        restore_rank_state(resume_state["rng_states"], rank, world_size, device, loader_generator)
        balance_state = resume_state["rng_states"][rank].get("extra", resume_state)
        ema_loss.update(balance_state["ema_loss"])
        balance_step = int(balance_state["balance_step"])
        if is_main:
            print(f"[VAE] Resumed at epoch {start_epoch+1}, optimizer step {global_step}")
        del resume_state
    if start_epoch >= epochs:
        raise ValueError("Checkpoint already completed the requested epochs; increase --epochs")

    # ── Epoch 0 validation (sanity check before training) ──
    synchronize()
    if is_main and vis_every > 0 and not args.resume_ckpt:
        print("  --- Validation (epoch 0, before training) ---", flush=True)
        vae_sd = vae
        validate_vae(vae_sd, cfg, -1, device, out_dir)  # epoch=-1 → prints as "epoch 0"

    synchronize()

    # ── Training loop ──
    t_start = time.time()
    for epoch in range(start_epoch, epochs):
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
            recon = decoder(z_patch.float())  # (B,1,D,H,W)

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

            window_size = min(accum_steps, len(loader) - (step // accum_steps) * accum_steps)
            (loss_g / window_size).backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
                opt_G.step()
                global_step += 1
                scheduler.step()
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
            scheduler.step()
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

            # Validation (every epoch, MUST be 1)
            if vis_every > 0 and (epoch + 1) % vis_every == 0:
                print(f"  --- Validation (epoch {epoch+1}) ---", flush=True)
                vae_sd = vae
                val_metrics = validate_vae(vae_sd, cfg, epoch, device, out_dir)
                print(f"  Val epoch {epoch+1}: ssim={val_metrics['ssim']:.4f} "
                      f"({val_metrics['n']} samples)", flush=True)

        synchronize()
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            rank_states = collect_rank_states(device, loader_generator, extra={
                "ema_loss": dict(ema_loss), "balance_step": balance_step,
            })
            if is_main:
                state = {
                    "state_dict": {f"autoencoder.{k}": v for k, v in vae.state_dict().items()},
                    "optimizer": opt_G.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "global_step": global_step, "config": cfg,
                    "ema_loss": dict(ema_loss), "balance_step": balance_step,
                    "rng_states": rank_states,
                    "steps_per_epoch": len(loader),
                }
                ckpt_path = out_dir / f"vae_finetune_epoch{epoch+1}.pt"
                atomic_save(state, ckpt_path)
                print(f"  Saved complete training state: {ckpt_path}", flush=True)
        synchronize()

    if is_main:
        vae_sd = vae
        atomic_save(
            {"state_dict": {f"autoencoder.{k}": v for k, v in vae_sd.state_dict().items()}},
            out_dir / "autoencoder_finetuned.pt",
        )
        print(f"Done. Drop-in: {out_dir}/autoencoder_finetuned.pt")

    if args.dist:
        dist.destroy_process_group()
