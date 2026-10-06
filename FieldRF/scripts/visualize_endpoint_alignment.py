"""Prospective validation for RF unpaired endpoint distribution alignment."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import PairedVisualizationDataset
from models.model import get_class_label
from train_endpoint_aligner import multiscale_patch_coral
from utils.visualization import (
    compute_metrics_eval,
    decode_volume_centered,
    eval_window_in_384,
    find_nifti_path,
    load_orig_ras_384,
    load_vae,
)


def _latent_metrics(prediction, target):
    prediction = prediction.float()
    target = target.float()
    residual = prediction - target
    denominator = torch.linalg.vector_norm(target.flatten()).clamp_min(1e-8)
    return {
        "latent_nrmse": (
            torch.linalg.vector_norm(residual.flatten()) / denominator
        ).item(),
        "latent_mae": residual.abs().mean().item(),
    }


def _append_dict_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _render(rows, title, path):
    if not rows:
        return
    figure, axes = plt.subplots(
        len(rows), 5, figsize=(19, 3.8 * len(rows)), squeeze=False
    )
    for index, row in enumerate(rows):
        images = [
            row["source"],
            row["baseline"],
            row["aligned"],
            row["target"],
            row["target_decoded"],
        ]
        middle = images[0].shape[2] // 2
        labels = [
            f"Source ({row['source_field']})",
            (
                "Baseline\n"
                f"SSIM={row['baseline_metrics']['ssim']:.3f} "
                f"nRMSE={row['baseline_metrics']['nrmse']:.3f}"
            ),
            (
                "Aligned\n"
                f"SSIM={row['aligned_metrics']['ssim']:.3f} "
                f"nRMSE={row['aligned_metrics']['nrmse']:.3f}"
            ),
            "Original Target (7T)",
            (
                "VAE Target\n"
                f"SSIM={row['vae_metrics']['ssim']:.3f}"
            ),
        ]
        for column, (image, label) in enumerate(zip(images, labels)):
            axes[index, column].imshow(
                image[:, :, middle],
                cmap="gray",
                vmin=0,
                vmax=1,
                origin="lower",
            )
            axes[index, column].set_title(label, fontsize=9)
            axes[index, column].axis("off")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def run_validation(bridge, aligner, cfg, epoch, device):
    """Evaluate endpoint and image changes on held-out prospective pairs."""
    validation_cfg = cfg.get("validation", {})
    data_cfg = cfg["data"]
    vae_cfg = cfg["vae"]
    output_dir = Path(cfg["output_dir"]) / "vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    split = validation_cfg.get("split", "Training_prospective")
    source_fields = validation_cfg.get(
        "source_fields", cfg["unpaired_alignment"]["source_fields"]
    )
    target_field = str(
        validation_cfg.get(
            "target_field",
            cfg["unpaired_alignment"]["reference_fields"][0],
        )
    )
    modalities = validation_cfg.get(
        "modalities",
        data_cfg.get("modalities", [data_cfg.get("modality", "T1W")]),
    )
    samples_per_field = int(validation_cfg.get("n_samples", 1))
    inversion_steps = int(
        validation_cfg.get(
            "inversion_steps",
            cfg["unpaired_alignment"].get("inversion_steps", 4),
        )
    )
    inversion_solver = str(
        validation_cfg.get(
            "inversion_solver",
            cfg["unpaired_alignment"].get("inversion_solver", "euler"),
        )
    )
    n_infer_steps = int(validation_cfg.get("n_infer_steps", 20))
    patch_sizes = cfg["unpaired_alignment"].get("patch_sizes", [12, 24])
    eval_range = tuple(validation_cfg.get("eval_slice_range", [150, 180]))
    scale_factor = float(bridge.scale_factor)
    amp_dtype = getattr(
        torch,
        vae_cfg.get("sliding_window", {}).get("amp_dtype", "float16"),
    )
    vae = load_vae(vae_cfg["checkpoint"], device)
    sw_cfg = vae_cfg["sliding_window"]

    bridge.eval()
    aligner.eval()
    case_rows = []
    summary_rows = []
    started = time.time()

    for modality in modalities:
        latent_root = (
            f"{data_cfg['cache_root']}/{split}/latent/{modality}"
        )
        feat_root = (
            f"{data_cfg['cache_root']}/{split}/"
            f"{data_cfg.get('feat_cache_name', 'brainid_feat')}/{modality}"
        )
        dataset = PairedVisualizationDataset(
            latent_root=latent_root,
            feat_all_root=feat_root,
            source_fields=source_fields,
            target_field=target_field,
            modality=modality,
            samples_per_field=samples_per_field,
        )
        modality_vis = []
        modality_rows = []

        for index in range(len(dataset)):
            sample = dataset[index]
            source_field = sample["source_field"]
            subject = sample["filename"].rsplit("_", 1)[-1]
            source_class = torch.tensor(
                [get_class_label(modality, source_field)],
                device=device,
            )
            target_class = torch.tensor(
                [get_class_label(modality, target_field)],
                device=device,
            )
            source_z = sample["z_src"].unsqueeze(0).to(device)
            target_z = sample["z_tgt"].unsqueeze(0).to(device)
            source_feats = [
                value.unsqueeze(0).to(device)
                for value in sample["feats_src"]
            ]
            target_feats = [
                value.unsqueeze(0).to(device)
                for value in sample["feats_tgt"]
            ]

            source_endpoint = bridge.invert_to_endpoint(
                source_z * scale_factor,
                source_feats,
                source_class,
                n_steps=inversion_steps,
                solver=inversion_solver,
            ).float()
            target_endpoint = bridge.invert_to_endpoint(
                target_z * scale_factor,
                target_feats,
                target_class,
                n_steps=inversion_steps,
                solver=inversion_solver,
            ).float()
            aligned_endpoint, delta = aligner(
                source_endpoint, source_class
            )
            endpoint_coral_before = multiscale_patch_coral(
                source_endpoint, target_endpoint, patch_sizes
            ).item()
            endpoint_coral_after = multiscale_patch_coral(
                aligned_endpoint.float(), target_endpoint, patch_sizes
            ).item()
            endpoint_l1_before = F.l1_loss(
                source_endpoint, target_endpoint
            ).item()
            endpoint_l1_after = F.l1_loss(
                aligned_endpoint.float(), target_endpoint
            ).item()

            baseline_latent = bridge.sample_x0_from_endpoint(
                source_endpoint,
                source_feats,
                target_class,
                n_infer_steps=n_infer_steps,
            )
            aligned_latent = bridge.sample_x0_from_endpoint(
                aligned_endpoint,
                source_feats,
                target_class,
                n_infer_steps=n_infer_steps,
            )
            baseline_latent_metrics = _latent_metrics(
                baseline_latent, target_z
            )
            aligned_latent_metrics = _latent_metrics(
                aligned_latent, target_z
            )

            target_nii = find_nifti_path(
                data_cfg["data_root"],
                split,
                modality,
                target_field,
                subject,
            )
            source_nii = find_nifti_path(
                data_cfg["data_root"],
                split,
                modality,
                source_field,
                subject,
            )
            if target_nii is None or source_nii is None:
                print(
                    f"  [val] SKIP image metrics {modality}/{source_field}/"
                    f"{subject}: source or target NIfTI missing",
                    flush=True,
                )
                continue

            original_target, _, pad_before_z, crop_start_z = (
                load_orig_ras_384(target_nii)
            )
            original_source, _, _, _ = load_orig_ras_384(source_nii)
            z0, z1 = eval_window_in_384(
                pad_before_z, crop_start_z, eval_range
            )
            z_center = (z0 + z1) // 2
            baseline_image = decode_volume_centered(
                vae, baseline_latent, sw_cfg, amp_dtype, z_center
            )
            aligned_image = decode_volume_centered(
                vae, aligned_latent, sw_cfg, amp_dtype, z_center
            )
            target_decoded_image = decode_volume_centered(
                vae, target_z, sw_cfg, amp_dtype, z_center
            )
            baseline_decoded = torch.clamp(
                (baseline_image + 1) / 2, 0, 1
            )[0, 0].float().cpu().numpy()
            aligned_decoded = torch.clamp(
                (aligned_image + 1) / 2, 0, 1
            )[0, 0].float().cpu().numpy()
            target_decoded = torch.clamp(
                (target_decoded_image + 1) / 2, 0, 1
            )[0, 0].float().cpu().numpy()

            window_start = round((z_center - 32) / 4) * 4
            crop_start = max(0, z0 - window_start)
            crop_end = min(baseline_decoded.shape[2], z1 - window_start)
            baseline_slab = baseline_decoded[:, :, crop_start:crop_end]
            aligned_slab = aligned_decoded[:, :, crop_start:crop_end]
            target_decoded_slab = target_decoded[
                :, :, crop_start:crop_end
            ]
            original_target_slab = original_target[:, :, z0:z1]
            original_source_slab = original_source[:, :, z0:z1]
            mask = (original_target_slab > 0).astype(np.float32)
            baseline_slab *= mask
            aligned_slab *= mask
            target_decoded_slab *= mask
            original_target_slab *= mask

            baseline_metrics = compute_metrics_eval(
                baseline_slab, original_target_slab, slice_axis=2
            )
            aligned_metrics = compute_metrics_eval(
                aligned_slab, original_target_slab, slice_axis=2
            )
            vae_metrics = compute_metrics_eval(
                target_decoded_slab, original_target_slab, slice_axis=2
            )
            row = {
                "epoch": epoch + 1,
                "modality": modality,
                "source_field": source_field,
                "target_field": target_field,
                "subject": subject,
                "endpoint_coral_before": endpoint_coral_before,
                "endpoint_coral_after": endpoint_coral_after,
                "endpoint_l1_before": endpoint_l1_before,
                "endpoint_l1_after": endpoint_l1_after,
                "endpoint_delta_rms": delta.float().square().mean().sqrt().item(),
                "baseline_latent_nrmse": baseline_latent_metrics[
                    "latent_nrmse"
                ],
                "aligned_latent_nrmse": aligned_latent_metrics[
                    "latent_nrmse"
                ],
                "baseline_latent_mae": baseline_latent_metrics["latent_mae"],
                "aligned_latent_mae": aligned_latent_metrics["latent_mae"],
                "baseline_image_nrmse": baseline_metrics["nrmse"],
                "aligned_image_nrmse": aligned_metrics["nrmse"],
                "baseline_image_ssim": baseline_metrics["ssim"],
                "aligned_image_ssim": aligned_metrics["ssim"],
                "baseline_image_psnr": baseline_metrics["psnr"],
                "aligned_image_psnr": aligned_metrics["psnr"],
                "vae_image_ssim": vae_metrics["ssim"],
            }
            case_rows.append(row)
            modality_rows.append(row)
            modality_vis.append(
                {
                    "source": original_source_slab,
                    "baseline": baseline_slab,
                    "aligned": aligned_slab,
                    "target": original_target_slab,
                    "target_decoded": target_decoded_slab,
                    "source_field": source_field,
                    "baseline_metrics": baseline_metrics,
                    "aligned_metrics": aligned_metrics,
                    "vae_metrics": vae_metrics,
                }
            )
            print(
                f"  [val] {modality} {source_field}->{target_field} "
                f"{subject}: endpoint CORAL "
                f"{endpoint_coral_before:.5f}->{endpoint_coral_after:.5f}, "
                f"SSIM {baseline_metrics['ssim']:.4f}->"
                f"{aligned_metrics['ssim']:.4f}",
                flush=True,
            )

        if modality_rows:
            numeric_keys = [
                key
                for key, value in modality_rows[0].items()
                if isinstance(value, (int, float)) and key != "epoch"
            ]
            summary = {
                "epoch": epoch + 1,
                "modality": modality,
                "n_cases": len(modality_rows),
            }
            for key in numeric_keys:
                summary[key] = float(
                    np.mean([float(row[key]) for row in modality_rows])
                )
            summary_rows.append(summary)
            _render(
                modality_vis,
                f"RF Unpaired Endpoint Alignment - {modality} - "
                f"Epoch {epoch+1}",
                output_dir
                / f"validation_epoch{epoch+1:03d}_{modality}.png",
            )

    _append_dict_rows(output_dir / "validation_metrics.csv", case_rows)
    _append_dict_rows(output_dir / "validation_summary.csv", summary_rows)
    aligned_ssims = [
        float(row["aligned_image_ssim"]) for row in case_rows
    ]
    baseline_ssims = [
        float(row["baseline_image_ssim"]) for row in case_rows
    ]
    result = {
        "aligned_image_ssim": (
            float(np.mean(aligned_ssims)) if aligned_ssims else float("nan")
        ),
        "baseline_image_ssim": (
            float(np.mean(baseline_ssims)) if baseline_ssims else float("nan")
        ),
        "n_cases": len(case_rows),
    }
    print(
        f"  [val] completed {result['n_cases']} cases in "
        f"{time.time()-started:.0f}s: mean SSIM "
        f"{result['baseline_image_ssim']:.4f}->"
        f"{result['aligned_image_ssim']:.4f}",
        flush=True,
    )
    aligner.train()
    return result
