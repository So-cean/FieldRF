"""Standalone evaluation script for MRIxFields2026 challenge.

Computes: nRMSE, SSIM, LPIPS, Dice (via SynthSeg), Volume Consistency.
Matches prediction-target pairs by subject ID prefix.

Task scope:
    Task 1 / Task 2: 5 metrics (nrmse + ssim + lpips + dice + volume).
    Task 3:         3 voxel-level metrics only (nrmse + ssim + lpips); no segmentations.

Usage:
    # Voxel-level metrics only (required path for Task 3)
    python Evaluation/evaluate.py \
        --pred_dir $INFERENCE_DIR/ \
        --target_dir $DATA_DIR/ground_truth/ \
        --metrics nrmse ssim lpips

    # Full evaluation (Task 1 / Task 2; requires pre-computed segmentations)
    python Evaluation/evaluate.py \
        --pred_dir $INFERENCE_DIR/ \
        --target_dir $DATA_DIR/ground_truth/ \
        --pred_seg_dir ${INFERENCE_DIR}_seg/ \
        --target_seg_dir $DATA_DIR/ground_truth_seg/ \
        --metrics nrmse ssim lpips dice volume \
        --output_csv results.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity


# --------------------------------------------------------------------------- #
#  NIfTI I/O
# --------------------------------------------------------------------------- #

def load_nifti(path):
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return img.get_fdata(dtype=np.float32), img.affine


def get_voxel_size(path):
    img = nib.load(str(path))
    return img.header.get_zooms()[:3]


# --------------------------------------------------------------------------- #
#  Voxel-level Metrics
# --------------------------------------------------------------------------- #

def compute_nrmse(pred, target, mask=None):
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    if mask is not None:
        pred, target = pred[mask > 0], target[mask > 0]
    norm = np.linalg.norm(target)
    return float(np.linalg.norm(pred - target) / norm) if norm > 1e-10 else 0.0


def compute_ssim(pred, target, slice_axis=2):
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    data_range = target.max() - target.min()
    if data_range < 1e-10:
        return 1.0
    if pred.ndim == 2:
        return float(structural_similarity(pred, target, data_range=data_range))
    vals = []
    for i in range(pred.shape[slice_axis]):
        s = [slice(None)] * pred.ndim
        s[slice_axis] = i
        s = tuple(s)
        ps, ts = pred[s], target[s]
        if ts.max() - ts.min() < 1e-10:
            continue
        vals.append(structural_similarity(ps, ts, data_range=data_range))
    return float(np.mean(vals)) if vals else 1.0


def compute_lpips(pred, target, slice_axis=2, device="cuda"):
    import torch
    import lpips as lpips_module
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    fn = lpips_module.LPIPS(net="alex").to(device)
    fn.eval()
    pred_n = pred.astype(np.float64) * 2.0 - 1.0
    target_n = target.astype(np.float64) * 2.0 - 1.0

    def _2d(p, t):
        pt = torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        tt = torch.from_numpy(t).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            return float(fn(pt, tt).item())

    if pred.ndim == 2:
        return _2d(pred_n, target_n)
    vals = []
    for i in range(pred.shape[slice_axis]):
        s = [slice(None)] * pred.ndim
        s[slice_axis] = i
        s = tuple(s)
        if np.abs(target_n[s]).max() < 1e-10:
            continue
        vals.append(_2d(pred_n[s], target_n[s]))
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
#  Segmentation-based Metrics (Dice & Volume)
# --------------------------------------------------------------------------- #

# 14 deep gray matter structures (7 bilateral pairs)
DGM_LABELS = {
    "L_Thalamus": 10,   "R_Thalamus": 49,
    "L_Caudate": 11,     "R_Caudate": 50,
    "L_Putamen": 12,     "R_Putamen": 51,
    "L_Pallidum": 13,    "R_Pallidum": 52,
    "L_Hippocampus": 17, "R_Hippocampus": 53,
    "L_Amygdala": 18,    "R_Amygdala": 54,
    "L_Accumbens": 26,   "R_Accumbens": 58,
}


def compute_dice(seg_pred, seg_target, labels=None):
    if labels is None:
        labels = DGM_LABELS
    scores = {}
    for name, lid in labels.items():
        pm = (seg_pred == lid)
        tm = (seg_target == lid)
        inter = np.sum(pm & tm)
        total = np.sum(pm) + np.sum(tm)
        scores[name] = 1.0 if total == 0 else float(2.0 * inter / total)
    return scores


def compute_volume_consistency(seg_pred, seg_target, voxel_volume=1.0, labels=None):
    if labels is None:
        labels = DGM_LABELS
    results = {}
    for name, lid in labels.items():
        vp = np.sum(seg_pred == lid) * voxel_volume
        vt = np.sum(seg_target == lid) * voxel_volume
        if vt < 1e-10:
            results[name] = 1.0 if vp < 1e-10 else 0.0
        else:
            results[name] = float(1.0 - abs(vp - vt) / vt)
    return results


# --------------------------------------------------------------------------- #
#  Matching & Pipeline
# --------------------------------------------------------------------------- #

def _extract_subject_id(filename):
    """Extract subject ID from filename for matching.

    Format: {R,P}_{modality}_{field}_{ID}.nii.gz -> ID (e.g., '0001')
    """
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return parts[-1]
    return parts[0]


def match_by_subject_prefix(pred_dir, target_dir):
    target_lookup = {}
    for f in Path(target_dir).rglob("*.nii.gz"):
        sid = _extract_subject_id(f.name)
        target_lookup[sid] = f
    pairs = []
    for pred_path in sorted(Path(pred_dir).rglob("*.nii.gz")):
        sid = _extract_subject_id(pred_path.name)
        if sid in target_lookup:
            pairs.append((pred_path, target_lookup[sid]))
    return pairs


def find_seg_file(seg_dir, subject_id, suffix="_seg"):
    """Find a segmentation file by subject ID.

    The filename must end with `{suffix}.nii.gz` (default `_seg.nii.gz`),
    matching the output convention of `segment.py`.
    """
    full_suffix = f"{suffix}.nii.gz"
    for f in Path(seg_dir).rglob(f"*{full_suffix}"):
        # Strip suffix so we can extract the subject ID from the original stem
        stem = f.name[: -len(full_suffix)]
        parts = stem.split("_")
        if parts and parts[-1] == subject_id:
            return f
    return None


def evaluate_pair(pred_path, target_path, metrics, device="cuda",
                  pred_seg_path=None, target_seg_path=None):
    pred, _ = load_nifti(pred_path)
    target, _ = load_nifti(target_path)

    # All voxel-level metrics (nRMSE / SSIM / LPIPS) are computed on the full
    # volume without a brain mask, to keep evaluation scope consistent.
    results = {}

    if "nrmse" in metrics:
        results["nrmse"] = compute_nrmse(pred, target)
    if "ssim" in metrics:
        results["ssim"] = compute_ssim(pred, target)
    if "lpips" in metrics:
        results["lpips"] = compute_lpips(pred, target, device=device)

    if "dice" in metrics or "volume" in metrics:
        seg_pred = nib.load(str(pred_seg_path)).get_fdata().astype(np.int32)
        seg_target = nib.load(str(target_seg_path)).get_fdata().astype(np.int32)

        if "dice" in metrics:
            scores = compute_dice(seg_pred, seg_target)
            results["dice"] = float(np.mean(list(scores.values())))

        if "volume" in metrics:
            # Use voxel size from the segmentation (SynthSeg resamples to 1mm iso)
            voxel_size = get_voxel_size(target_seg_path)
            voxel_vol = float(np.prod(voxel_size))
            scores = compute_volume_consistency(seg_pred, seg_target, voxel_vol)
            results["volume"] = float(np.mean(list(scores.values())))

    return results


def main():
    parser = argparse.ArgumentParser(description="MRIxFields2026 Evaluation")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--pred_seg_dir", type=str, default=None,
                        help="Directory of SynthSeg segmentations for predictions (required for dice/volume)")
    parser.add_argument("--target_seg_dir", type=str, default=None,
                        help="Directory of SynthSeg segmentations for ground truth (required for dice/volume)")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--metrics", nargs="+", default=["nrmse", "ssim", "lpips"],
                        choices=["nrmse", "ssim", "lpips", "dice", "volume"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seg_suffix", type=str, default="_seg",
                        help="Suffix used by segment.py outputs (default: _seg)")
    args = parser.parse_args()

    # Validate: dice/volume require seg dirs
    need_seg = "dice" in args.metrics or "volume" in args.metrics
    if need_seg:
        if args.pred_seg_dir is None or args.target_seg_dir is None:
            parser.error(
                "Dice/Volume metrics require --pred_seg_dir and --target_seg_dir.\n"
                "Run SynthSeg segmentation first:\n"
                "  python Evaluation/segment.py --input_dir <pred_dir> --output_dir <pred_seg_dir>\n"
                "  python Evaluation/segment.py --input_dir <target_dir> --output_dir <target_seg_dir>"
            )

    pairs = match_by_subject_prefix(args.pred_dir, args.target_dir)
    if not pairs:
        print("No matching pairs found.")
        return

    all_results = []
    for pred_path, target_path in pairs:
        subject = _extract_subject_id(pred_path.name)
        print(f"Evaluating: {subject} ({pred_path.name} <-> {target_path.name})")

        pred_seg_path = None
        target_seg_path = None
        if need_seg:
            pred_seg_path = find_seg_file(args.pred_seg_dir, subject, args.seg_suffix)
            target_seg_path = find_seg_file(args.target_seg_dir, subject, args.seg_suffix)
            if pred_seg_path is None or target_seg_path is None:
                print(f"  ERROR: Segmentation file not found for subject {subject}.")
                print(f"  Run: python Evaluation/segment.py --input_dir ... --output_dir ...")
                continue

        result = evaluate_pair(
            pred_path, target_path, args.metrics, args.device,
            pred_seg_path=pred_seg_path,
            target_seg_path=target_seg_path,
        )
        result["subject"] = subject
        all_results.append(result)

    if not all_results:
        print("No results computed.")
        return

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            metric_cols = [k for k in all_results[0] if k != "subject"]
            w = csv.DictWriter(f, fieldnames=["subject"] + metric_cols)
            w.writeheader()
            w.writerows(all_results)
        print(f"Saved: {args.output_csv}")

    metric_keys = [k for k in all_results[0] if k != "subject"]
    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in all_results]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

    print(f"\n{'='*50}")
    print("Evaluation Summary")
    print(f"{'='*50}")
    for k in metric_keys:
        d = "lower" if k in ["nrmse", "lpips"] else "higher"
        print(f"  {k:>10s}: {summary[f'{k}_mean']:.4f} +/- {summary[f'{k}_std']:.4f}  ({d} is better)")
    print(f"{'='*50}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
