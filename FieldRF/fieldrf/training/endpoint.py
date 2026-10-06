"""Train an unpaired distribution aligner for frozen RF inversion endpoints.

Only real retrospective caches are used.  A source sample and an independent
same-modality reference-field sample are inverted by a frozen Stage-1 RF.
The zero-initialized residual adapter is optimized with patch-distribution
CORAL; no sample-wise endpoint, image, cycle, or synthetic-pair loss is used.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from fieldrf.data.alignment_dataset import UnpairedEndpointAlignmentDataset
from fieldrf.models.model import FieldRFModel  # noqa: E402
from fieldrf.utils.domains import NUM_CLASSES  # noqa: E402
from fieldrf.models.endpoint_aligner import EndpointAligner
from fieldrf.utils.endpoint_alignment import multiscale_patch_coral


def spatial_delta_loss(delta):
    """Penalize high-frequency corrections while preserving identity at init."""
    gradients = []
    for axis in (2, 3, 4):
        leading = [slice(None)] * delta.ndim
        trailing = [slice(None)] * delta.ndim
        leading[axis] = slice(1, None)
        trailing[axis] = slice(None, -1)
        gradients.append(
            (delta[tuple(leading)] - delta[tuple(trailing)])
            .float()
            .square()
            .mean()
        )
    return torch.stack(gradients).mean()


def _build_bridge(cfg, checkpoint, device):
    bridge_cfg = cfg["bridge"]
    bridge = FieldRFModel.from_pretrained(
        checkpoint,
        unet_channels=tuple(
            bridge_cfg.get("unet_channels", [64, 128, 256, 512])
        ),
        num_res_blocks=tuple(
            bridge_cfg.get("num_res_blocks", [2, 2, 2, 2])
        ),
        attention_levels=tuple(
            bridge_cfg.get("attention_levels", [False, False, True, True])
        ),
        num_head_channels=tuple(
            bridge_cfg.get("num_head_channels", [0, 0, 32, 32])
        ),
        norm_num_groups=bridge_cfg.get("norm_num_groups", 32),
        norm_eps=bridge_cfg.get("norm_eps", 1e-6),
        num_class_embeds=NUM_CLASSES,
        feat4_channels=bridge_cfg.get("feat4_channels", 64),
        z_channels=4,
        num_train_timesteps=bridge_cfg.get("num_train_timesteps", 1000),
        scale_factor=float(cfg["vae"].get("scale_factor", 1.0)),
        use_source_latent=False,
        device=device,
        load_ema=True,
    )
    bridge.eval()
    bridge.requires_grad_(False)
    return bridge


def _append_log(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run(args):

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    from fieldrf.training.distributed import setup_distributed, wrap_module
    rank, world_size, local_rank, device = setup_distributed(args.dist, args.device)
    is_main = rank == 0

    data_cfg = cfg["data"]
    align_cfg = cfg["unpaired_alignment"]
    train_cfg = cfg["train"]
    modalities = data_cfg.get("modalities", [data_cfg["modality"]])
    dataset = UnpairedEndpointAlignmentDataset(
        cache_root=data_cfg["cache_root"],
        split=data_cfg.get("split", "Training_retrospective"),
        modalities=modalities,
        source_fields=align_cfg["source_fields"],
        reference_fields=align_cfg["reference_fields"],
        feat_cache_name=data_cfg.get("feat_cache_name", "brainid_feat"),
        max_volumes=args.max_volumes,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if args.dist else None
    num_workers = int(train_cfg.get("num_workers", 4))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(
            train_cfg.get("prefetch_factor", 2)
        )
    loader = DataLoader(**loader_kwargs)

    base_checkpoint = cfg.get("finetune_from")
    if not base_checkpoint:
        raise ValueError(
            "Set finetune_from to a trained RF Stage-1 checkpoint."
        )
    bridge = _build_bridge(cfg, base_checkpoint, device)
    model_cfg = align_cfg.get("model", {})
    aligner = EndpointAligner(
        endpoint_channels=4,
        hidden_channels=model_cfg.get("hidden_channels", 24),
        domain_embed_dim=model_cfg.get("domain_embed_dim", 8),
        num_domains=NUM_CLASSES,
        num_blocks=model_cfg.get("num_blocks", 2),
    ).to(device)
    if args.dist:
        aligner = wrap_module(aligner, device)
    aligner_raw = aligner.module if args.dist else aligner

    output_dir = Path(
        cfg.get(
            "output_dir", "runs/endpoint_alignment"
        )
    )
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            from fieldrf.utils.visualization import archive_metrics_csvs

            archive_metrics_csvs(
                output_dir / "vis",
                patterns=["validation_*.csv"],
            )
            archive_metrics_csvs(
                output_dir,
                patterns=["training_log.csv"],
            )

    n_epochs = int(args.n_epochs or train_cfg.get("n_epochs", 20))
    total_steps = max(1, n_epochs * len(loader))
    warmup_steps = int(
        float(train_cfg.get("warmup_epochs", 1)) * len(loader)
    )
    lr = float(train_cfg.get("lr", 1e-4))
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    optimizer = torch.optim.AdamW(
        aligner_raw.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min / lr + (1.0 - lr_min / lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch = 0
    global_step = 0
    best_validation_ssim = float("-inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        aligner_raw.load_state_dict(state["endpoint_aligner"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", start_epoch * len(loader)))
        best_validation_ssim = float(
            state.get("best_validation_ssim", float("-inf"))
        )

    amp_dtype = getattr(torch, train_cfg.get("mixed_precision", "bfloat16"))
    autocast_enabled = device.type == "cuda"
    inversion_steps = int(align_cfg.get("inversion_steps", 4))
    inversion_solver = str(align_cfg.get("inversion_solver", "euler"))
    patch_sizes = align_cfg.get("patch_sizes", [12, 24])
    lambda_align = float(align_cfg.get("lambda_align", 1.0))
    lambda_delta = float(align_cfg.get("lambda_delta", 0.01))
    lambda_spatial = float(align_cfg.get("lambda_spatial", 0.0))
    lambda_reference_identity = float(
        align_cfg.get("lambda_reference_identity", 0.01)
    )
    scale_factor = float(bridge.scale_factor)

    if is_main:
        print(
            f"Unpaired real endpoint alignment: samples={len(dataset)} "
            f"modalities={modalities} source={align_cfg['source_fields']} "
            f"reference={align_cfg['reference_fields']}",
            flush=True,
        )
        print(
            "Frozen Stage-1 RF + zero-initialized residual adapter; "
            f"CORAL patches={patch_sizes}, inversion="
            f"{inversion_solver}-{inversion_steps}",
            flush=True,
        )

    for epoch in range(start_epoch, n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        aligner.train()
        sums = {
            "total": 0.0,
            "align": 0.0,
            "delta": 0.0,
            "spatial": 0.0,
            "reference_identity": 0.0,
        }
        started = time.time()

        for step, batch in enumerate(loader):
            source_z = batch["z_src"].to(device) * scale_factor
            reference_z = batch["z_ref"].to(device) * scale_factor
            source_feats = [value.to(device) for value in batch["feats_src"]]
            reference_feats = [
                value.to(device) for value in batch["feats_ref"]
            ]
            source_class = batch["source_class"].to(device)
            reference_class = batch["reference_class"].to(device)

            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=autocast_enabled, dtype=amp_dtype
            ):
                source_endpoint = bridge.invert_to_endpoint(
                    source_z,
                    source_feats,
                    source_class,
                    n_steps=inversion_steps,
                    solver=inversion_solver,
                ).float()
                reference_endpoint = bridge.invert_to_endpoint(
                    reference_z,
                    reference_feats,
                    reference_class,
                    n_steps=inversion_steps,
                    solver=inversion_solver,
                ).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=autocast_enabled, dtype=amp_dtype
            ):
                combined_endpoint = torch.cat(
                    [source_endpoint, reference_endpoint], dim=0
                )
                combined_class = torch.cat(
                    [source_class, reference_class], dim=0
                )
                combined_aligned, combined_delta = aligner(
                    combined_endpoint, combined_class
                )
                source_batch = source_endpoint.shape[0]
                aligned_endpoint = combined_aligned[:source_batch]
                reference_aligned = combined_aligned[source_batch:]
                delta = combined_delta[:source_batch]
                alignment_loss = multiscale_patch_coral(
                    aligned_endpoint.float(),
                    reference_endpoint,
                    patch_sizes,
                )
                delta_loss = delta.float().square().mean()
                spatial_loss = spatial_delta_loss(delta)
                reference_identity_loss = F.mse_loss(
                    reference_aligned.float(), reference_endpoint
                )
                total_loss = (
                    lambda_align * alignment_loss
                    + lambda_delta * delta_loss
                    + lambda_spatial * spatial_loss
                    + lambda_reference_identity * reference_identity_loss
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                aligner_raw.parameters(),
                float(train_cfg.get("grad_clip", 1.0)),
            )
            optimizer.step()
            scheduler.step()

            values = {
                "total": total_loss.detach().item(),
                "align": alignment_loss.detach().item(),
                "delta": delta_loss.detach().item(),
                "spatial": spatial_loss.detach().item(),
                "reference_identity": reference_identity_loss.detach().item(),
            }
            for key, value in values.items():
                sums[key] += value
            global_step += 1

            log_every = int(train_cfg.get("log_every", 25))
            if is_main and (step + 1) % log_every == 0:
                print(
                    f"[{epoch+1}/{n_epochs}] {step+1}/{len(loader)} "
                    f"loss={values['total']:.6f} "
                    f"align={values['align']:.6f} "
                    f"delta={values['delta']:.6f} "
                    f"spatial={values['spatial']:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        means = {key: value / max(1, len(loader)) for key, value in sums.items()}
        if args.dist:
            dist.barrier()
        validation_result = None
        validation_cfg = cfg.get("validation", {})
        validation_every = int(validation_cfg.get("every_epochs", 1))
        if (
            is_main
            and validation_every > 0
            and (epoch + 1) % validation_every == 0
        ):
            from fieldrf.visualization.endpoint import run_validation

            validation_result = run_validation(
                bridge,
                aligner_raw,
                cfg,
                epoch,
                device,
            )
        if args.dist:
            dist.barrier()

        if is_main:
            row = {
                "epoch": epoch + 1,
                "global_step": global_step,
                **means,
                "val_baseline_ssim": (
                    validation_result["baseline_image_ssim"]
                    if validation_result is not None
                    else float("nan")
                ),
                "val_aligned_ssim": (
                    validation_result["aligned_image_ssim"]
                    if validation_result is not None
                    else float("nan")
                ),
                "val_n_cases": (
                    validation_result["n_cases"]
                    if validation_result is not None
                    else 0
                ),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - started,
            }
            _append_log(output_dir / "training_log.csv", row)
            print(
                f"Epoch {epoch+1}: loss={means['total']:.6f} "
                f"align={means['align']:.6f} "
                f"delta={means['delta']:.6f} "
                f"spatial={means['spatial']:.6f}",
                flush=True,
            )

            validation_ssim = (
                float(validation_result["aligned_image_ssim"])
                if validation_result is not None
                else float("nan")
            )
            is_best = (
                math.isfinite(validation_ssim)
                and validation_ssim > best_validation_ssim
            )
            if is_best:
                best_validation_ssim = validation_ssim

            def checkpoint_state():
                return {
                    "endpoint_aligner": aligner_raw.state_dict(),
                    "endpoint_aligner_config": model_cfg,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config": cfg,
                    "base_checkpoint": str(base_checkpoint),
                    "best_validation_ssim": best_validation_ssim,
                    "validation": validation_result,
                    "training_objective": (
                        "unpaired_endpoint_distribution_alignment"
                    ),
                }

            save_every = int(train_cfg.get("save_every", 1))
            if (epoch + 1) % save_every == 0 or epoch + 1 == n_epochs:
                checkpoint_path = (
                    output_dir
                    / f"fieldrf_unpaired_alignment_epoch{epoch+1}.pt"
                )
                torch.save(checkpoint_state(), checkpoint_path)
                print(f"Saved {checkpoint_path}", flush=True)
            if is_best:
                best_path = (
                    output_dir / "fieldrf_unpaired_alignment_best.pt"
                )
                torch.save(checkpoint_state(), best_path)
                print(
                    f"Saved best validation checkpoint: {best_path} "
                    f"(SSIM={best_validation_ssim:.4f})",
                    flush=True,
                )

    if args.dist:
        dist.destroy_process_group()
