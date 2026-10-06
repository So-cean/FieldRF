"""Same-modality Any-to-Any visualization for RF paired fine-tuning.

The per-epoch visualization uses one fixed prospective training subject and
materializes every same-modality ordered field pair. Short
Euler trajectories make this a trend monitor rather than the final Euler-20
challenge evaluation.  Every generated image is nevertheless decoded and
scored in the official central 30-slice window against its same-subject real
target.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


FIELDRF_ROOT = Path(__file__).resolve().parents[2]

from fieldrf.data.dataset import (  # noqa: E402
    PairedFieldDataset,
    load_feat4,
    load_latent,
)
from fieldrf.utils.domains import get_class_label  # noqa: E402
from fieldrf.utils.visualization import (  # noqa: E402
    compute_metrics_eval,
    decode_volume_centered,
    eval_window_in_384,
    find_nifti_path,
    load_orig_ras_384,
    load_vae,
)


def _domain_label(modality, field):
    return f"{modality}@{field}"


def _append_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _safe_percentile(matrix, percentile, default):
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return float(default)
    return float(np.percentile(finite, percentile))


def _render_metric_matrix(rows, domains, epoch, subject, output_path):
    domain_to_index = {domain: index for index, domain in enumerate(domains)}
    matrices = {
        "SSIM ↑": np.full((len(domains), len(domains)), np.nan, np.float32),
        "nRMSE ↓": np.full((len(domains), len(domains)), np.nan, np.float32),
        "PSNR ↑": np.full((len(domains), len(domains)), np.nan, np.float32),
    }
    for row in rows:
        source = domain_to_index[row["source_domain"]]
        target = domain_to_index[row["target_domain"]]
        matrices["SSIM ↑"][source, target] = float(row["ssim_image"])
        matrices["nRMSE ↓"][source, target] = float(row["nrmse_image"])
        matrices["PSNR ↑"][source, target] = float(row["psnr_image"])

    figure, axes = plt.subplots(1, 3, figsize=(30, 9), constrained_layout=True)
    specifications = (
        ("SSIM ↑", "viridis", 0.0, 1.0),
        (
            "nRMSE ↓",
            "magma_r",
            0.0,
            _safe_percentile(matrices["nRMSE ↓"], 95, 1.0),
        ),
        (
            "PSNR ↑",
            "viridis",
            _safe_percentile(matrices["PSNR ↑"], 5, 0.0),
            _safe_percentile(matrices["PSNR ↑"], 95, 40.0),
        ),
    )
    for axis, (name, colour_map, minimum, maximum) in zip(axes, specifications):
        matrix = matrices[name]
        image = axis.imshow(
            matrix,
            cmap=colour_map,
            vmin=minimum,
            vmax=max(maximum, minimum + 1.0e-6),
        )
        axis.set_xticks(range(len(domains)), labels=domains, rotation=90)
        axis.set_yticks(range(len(domains)), labels=domains)
        axis.set_xlabel("target domain")
        axis.set_ylabel("source domain")
        axis.set_title(name)
        for source in range(len(domains)):
            for target in range(len(domains)):
                value = matrix[source, target]
                axis.text(
                    target,
                    source,
                    "—" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) else "gray",
                    fontsize=4.5,
                )
        figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    figure.suptitle(
        f"RF paired fine-tune 15-domain Any-to-Any — epoch {epoch}, subject {subject}"
    )
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _render_image_matrix(images, scores, domains, epoch, subject, output_path):
    figure, axes = plt.subplots(
        len(domains),
        len(domains),
        figsize=(30, 30),
        constrained_layout=True,
    )
    for source, source_domain in enumerate(domains):
        for target, target_domain in enumerate(domains):
            axis = axes[source, target]
            key = (source_domain, target_domain)
            if key in images:
                axis.imshow(images[key], cmap="gray", vmin=0, vmax=1)
                axis.set_title(f"{scores[key]:.3f}", fontsize=5)
            else:
                # Task3 paired fine-tuning intentionally evaluates only
                # same-modality cross-field blocks.  Keep the undefined
                # cross-modality cells explicit instead of indexing them.
                axis.set_facecolor("black")
                axis.text(
                    0.5,
                    0.5,
                    "N/A",
                    color="gray",
                    fontsize=5,
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if source == 0:
                axis.set_xlabel(target_domain, fontsize=6)
                axis.xaxis.set_label_position("top")
            if target == 0:
                axis.set_ylabel(source_domain, fontsize=6)
    figure.suptitle(
        f"Generated center slices; title=SSIM — epoch {epoch}, subject {subject}",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _summarize(rows):
    if not rows:
        return {"cases": 0, "ssim": None, "nrmse": None, "psnr": None,
                "bridge_ssim": None}
    return {
        "cases": len(rows),
        "ssim": float(np.mean([float(row["ssim_image"]) for row in rows])),
        "nrmse": float(np.mean([float(row["nrmse_image"]) for row in rows])),
        "psnr": float(np.mean([float(row["psnr_image"]) for row in rows])),
        "bridge_ssim": float(np.mean([float(row["ssim_bridge"]) for row in rows])),
    }


@torch.no_grad()
def run_vis(
    bridge,
    scale_factor,
    config,
    epoch,
    device,
    endpoint_aligner=None,
):
    """Run one complete 15-domain matrix and return its aggregate metrics."""
    data = config["data"]
    vis = config["vis"]
    vae_config = config["vae"]
    modalities = tuple(str(value) for value in data["modalities"])
    fields = tuple(str(value) for value in data["fields"])
    domains = tuple(_domain_label(*domain) for domain in product(modalities, fields))
    subject = str(vis.get("any_to_any_subject", data["subjects"][0]))
    if subject not in {str(value) for value in data["subjects"]}:
        raise ValueError(f"Any-to-Any vis subject is not trained: {subject}")
    inversion_steps = int(vis.get("any_to_any_inversion_steps", 4))
    generation_steps = int(vis.get("any_to_any_generation_steps", 4))
    eval_range = tuple(vis.get("eval_slice_range", [150, 180]))
    amp_dtype = getattr(
        torch,
        vae_config.get("sliding_window", {}).get("amp_dtype", "float16"),
    )
    sliding_window = vae_config.get("sliding_window", {"roi_size": [384, 384, 64]})
    run_root = Path(config["output_dir"])
    if not run_root.is_absolute():
        run_root = FIELDRF_ROOT / run_root
    output_dir = run_root / "vis_any_to_any"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_vis_contract.json").write_text(
        json.dumps(
            {
                "type": "real_paired_any_to_any",
                "subject": subject,
                "modalities": list(modalities),
                "fields": list(fields),
                "ordered_cases": 225,
                "inversion_steps": inversion_steps,
                "generation_steps": generation_steps,
                "evaluation_slice_range": list(eval_range),
                "checkpoint_selection": False,
                "final_euler20_evaluation_required": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    dataset = PairedFieldDataset(
        cache_root=data["cache_root"],
        split=vis.get("split", data["split"]),
        modalities=modalities,
        source_fields=fields,
        target_fields=fields,
        subjects=[subject],
        feat_cache_name=data["feat_cache_name"],
        include_identity=True,
    )
    grouped = defaultdict(list)
    for record in dataset.records:
        grouped[(record["source_modality"], record["source_field"])].append(record)
    expected_pairs = len(modalities) * len(fields) * len(fields)
    if len(dataset) != expected_pairs or len(grouped) != 15:
        raise RuntimeError(
            f"Expected {expected_pairs} same-modality pairs, got {len(dataset)} records "
            f"and {len(grouped)} source domains"
        )

    vae = load_vae(vae_config["checkpoint"], device)
    target_cache = {}
    for target_modality, target_field in product(modalities, fields):
        target_path = find_nifti_path(
            data["data_root"],
            vis.get("split", data["split"]),
            target_modality,
            target_field,
            subject,
        )
        if target_path is None:
            raise FileNotFoundError(
                f"Missing target NIfTI {target_modality}@{target_field}/{subject}"
            )
        target_volume, _, pad_before, crop_start = load_orig_ras_384(target_path)
        scored_start, scored_end = eval_window_in_384(
            pad_before, crop_start, eval_range
        )
        center = (scored_start + scored_end) // 2
        target_scored = target_volume[..., scored_start:scored_end]
        mask = (target_scored > 0).astype(np.float32)
        stem = f"P_{target_modality}_{target_field}_{subject}"
        target_latent_path = (
            Path(data["cache_root"])
            / vis.get("split", data["split"])
            / "latent"
            / target_modality
            / target_field
            / f"{stem}.pt"
        )
        target_latent = load_latent(target_latent_path).unsqueeze(0).to(device)
        target_decoded = decode_volume_centered(
            vae, target_latent, sliding_window, amp_dtype, center
        )
        target_decoded = (
            ((target_decoded + 1.0) / 2.0).clamp(0.0, 1.0)[0, 0].float().cpu().numpy()
        )
        decoded_start = round((center - 32) / 4) * 4
        local_start = scored_start - decoded_start
        local_end = scored_end - decoded_start
        target_cache[(target_modality, target_field)] = {
            "center": center,
            "local_start": local_start,
            "local_end": local_end,
            "original": target_scored * mask,
            "decoded": target_decoded[..., local_start:local_end] * mask,
            "mask": mask,
        }
        del target_latent, target_decoded

    bridge.eval()
    rows = []
    center_images = {}
    center_scores = {}
    started = time.time()
    for source_index, ((source_modality, source_field), records) in enumerate(
        sorted(grouped.items())
    ):
        source_record = records[0]
        source_latent = (
            load_latent(source_record["source_latent"]).unsqueeze(0).to(device)
        )
        source_feat4 = (
            load_feat4(source_record["source_feature"]).unsqueeze(0).to(device)
        )
        source_class = torch.tensor(
            [get_class_label(source_modality, source_field)],
            device=device,
            dtype=torch.long,
        )
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            source_endpoint = bridge.invert_to_endpoint(
                source_latent.float() * scale_factor,
                [source_feat4],
                source_class,
                n_steps=inversion_steps,
                solver="euler",
                gradient_steps=0,
                velocity_model=bridge,
            )
            if endpoint_aligner is not None:
                source_endpoint, _ = endpoint_aligner(
                    source_endpoint.float(), source_class
                )

        for record in sorted(
            records,
            key=lambda value: (
                modalities.index(value["target_modality"]),
                fields.index(value["target_field"]),
            ),
        ):
            target_modality = record["target_modality"]
            target_field = record["target_field"]
            target = target_cache[(target_modality, target_field)]
            target_class = torch.tensor(
                [get_class_label(target_modality, target_field)],
                device=device,
                dtype=torch.long,
            )
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                generated_latent = bridge.sample_x0_from_endpoint(
                    source_endpoint,
                    [source_feat4],
                    target_class,
                    n_infer_steps=generation_steps,
                )
                generated_image = decode_volume_centered(
                    vae,
                    generated_latent,
                    sliding_window,
                    amp_dtype,
                    target["center"],
                )
            generated_image = (
                ((generated_image + 1.0) / 2.0)
                .clamp(0.0, 1.0)[0, 0]
                .float()
                .cpu()
                .numpy()
            )
            generated_scored = (
                generated_image[..., target["local_start"] : target["local_end"]]
                * target["mask"]
            )
            image_metrics = compute_metrics_eval(
                generated_scored, target["original"], slice_axis=2
            )
            bridge_metrics = compute_metrics_eval(
                generated_scored, target["decoded"], slice_axis=2
            )
            source_domain = _domain_label(source_modality, source_field)
            target_domain = _domain_label(target_modality, target_field)
            row = {
                "epoch": epoch + 1,
                "subject": subject,
                "source_modality": source_modality,
                "source_field": source_field,
                "target_modality": target_modality,
                "target_field": target_field,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "ssim_image": float(image_metrics["ssim"]),
                "nrmse_image": float(image_metrics["nrmse"]),
                "psnr_image": float(image_metrics["psnr"]),
                "ssim_bridge": float(bridge_metrics["ssim"]),
                "nrmse_bridge": float(bridge_metrics["nrmse"]),
                "psnr_bridge": float(bridge_metrics["psnr"]),
            }
            rows.append(row)
            center = generated_scored.shape[-1] // 2
            center_images[(source_domain, target_domain)] = generated_scored[
                ..., center
            ]
            center_scores[(source_domain, target_domain)] = row["ssim_image"]
            del generated_latent, generated_image, generated_scored

        print(
            f"  [any2any vis] source {source_index + 1}/15 "
            f"{source_modality}@{source_field} complete",
            flush=True,
        )
        del source_latent, source_feat4, source_endpoint

    metrics_path = output_dir / "metrics.csv"
    _append_metrics(metrics_path, rows)
    _render_metric_matrix(
        rows,
        domains,
        epoch + 1,
        subject,
        output_dir / f"epoch{epoch + 1:03d}_metric_matrix.png",
    )
    if bool(vis.get("render_image_matrix", True)):
        _render_image_matrix(
            center_images,
            center_scores,
            domains,
            epoch + 1,
            subject,
            output_dir / f"epoch{epoch + 1:03d}_image_matrix.png",
        )

    identity = [row for row in rows if row["source_domain"] == row["target_domain"]]
    same_modality_cross_field = [
        row
        for row in rows
        if row["source_modality"] == row["target_modality"]
        and row["source_field"] != row["target_field"]
    ]
    cross_modality = [
        row for row in rows if row["source_modality"] != row["target_modality"]
    ]
    non_identity = [row for row in rows if row["source_domain"] != row["target_domain"]]
    summary = {
        "epoch": epoch + 1,
        "subject": subject,
        "inversion_steps": inversion_steps,
        "generation_steps": generation_steps,
        "all": _summarize(rows),
        "non_identity": _summarize(non_identity),
        "identity": _summarize(identity),
        "same_modality_cross_field": _summarize(same_modality_cross_field),
        "cross_modality": _summarize(cross_modality),
        "seconds": time.time() - started,
    }
    (output_dir / f"epoch{epoch + 1:03d}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    bridge.train()
    print(
        "  [any2any vis] "
        f"non_identity_ssim={summary['non_identity']['ssim']:.5f} "
        f"same_mod_cross_field={summary['same_modality_cross_field']['ssim']:.5f} "
        f"identity={summary['identity']['ssim']:.5f} "
        f"seconds={summary['seconds']:.0f}",
        flush=True,
    )
    return summary


__all__ = ["run_vis"]
