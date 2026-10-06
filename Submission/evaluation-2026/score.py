#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MRIxFields2026 — Validation Scorer.

Tasks:
  - task1: Any -> 7T            (4 pairs x 3 modalities, 5 metrics)
  - task2: 0.1T -> Higher       (4 pairs x 3 modalities, 5 metrics)
  - task3: Any -> Any (unified) (20 pairs x 3 modalities, 3 voxel-level metrics)

Metrics:
  - nRMSE / SSIM / LPIPS  : voxel-level, all tasks
  - Dice / Volume         : segmentation-based (DGM 14 labels), task1 / task2 only

Inputs (all three must agree on the task):
  -i / --input    : directory whose last path component is `task{N}` and that
                    contains `{T1W,T2W,T2FLAIR}/{pair}/{pred,seg}/...` directly.
  -g / --gt_dir   : directory whose last path component is `task{N}` and that
                    contains `{T1W,T2W,T2FLAIR}/{pair}/{gt,gt_seg}/...` directly.
  -t / --task     : task1 / task2 / task3 — must equal both directories' last
                    path component (cross-validation).

This scorer does not extract archives. Unzip ahead of time and pass the
directory.

Layouts:
    {input}/{T1W,T2W,T2FLAIR}/{pair}/pred/P_{MOD}_{TGT}_{ID:04d}.nii.gz
    {input}/{T1W,T2W,T2FLAIR}/{pair}/seg/ P_{MOD}_{TGT}_{ID:04d}_seg.nii.gz   (task1/task2 only)

    {gt_dir}/{T1W,T2W,T2FLAIR}/{pair}/gt/    P_{MOD}_{TGT}_{ID:04d}.nii.gz
    {gt_dir}/{T1W,T2W,T2FLAIR}/{pair}/gt_seg/P_{MOD}_{TGT}_{ID:04d}_seg.nii.gz   (task1/task2 only)

The set of expected subjects per (modality, pair) is whatever IDs exist
under that pair's `gt/` directory — the GT tree itself defines what a
complete submission looks like.

Output:
    {output}/Result/results.json    — primary, consumed by the platform
    {output}/Result/result_{task}.xlsx — per-sample detail + summary sheets
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Task configuration
# ============================================================

MODALITIES: List[str] = ["T1W", "T2W", "T2FLAIR"]

TASK_CONFIG: Dict[str, Dict] = {
    "task1": {
        "pairs": ["0.1T_to_7T", "1.5T_to_7T", "3T_to_7T", "5T_to_7T"],
        "metrics": ["nRMSE", "SSIM", "LPIPS", "Dice", "Volume"],
        "needs_seg": True,
    },
    "task2": {
        "pairs": ["0.1T_to_1.5T", "0.1T_to_3T", "0.1T_to_5T", "0.1T_to_7T"],
        "metrics": ["nRMSE", "SSIM", "LPIPS", "Dice", "Volume"],
        "needs_seg": True,
    },
    "task3": {
        # Any -> Any across all 5 fields (5 x 4 = 20 directed pairs, incl. downfield).
        "pairs": [
            "0.1T_to_1.5T", "0.1T_to_3T",  "0.1T_to_5T", "0.1T_to_7T",
            "1.5T_to_0.1T", "1.5T_to_3T",  "1.5T_to_5T", "1.5T_to_7T",
            "3T_to_0.1T",   "3T_to_1.5T",  "3T_to_5T",   "3T_to_7T",
            "5T_to_0.1T",   "5T_to_1.5T",  "5T_to_3T",   "5T_to_7T",
            "7T_to_0.1T",   "7T_to_1.5T",  "7T_to_3T",   "7T_to_5T",
        ],
        "metrics": ["nRMSE", "SSIM", "LPIPS"],
        "needs_seg": False,
    },
}

# 14 deep gray-matter structures (FreeSurfer / SynthSeg label IDs).
DGM_LABELS: Dict[str, int] = {
    "L_Thalamus": 10,    "R_Thalamus": 49,
    "L_Caudate": 11,     "R_Caudate": 50,
    "L_Putamen": 12,     "R_Putamen": 51,
    "L_Pallidum": 13,    "R_Pallidum": 52,
    "L_Hippocampus": 17, "R_Hippocampus": 53,
    "L_Amygdala": 18,    "R_Amygdala": 54,
    "L_Accumbens": 26,   "R_Accumbens": 58,
}

PAIR_RE = re.compile(r"^(?P<src>[0-9.]+T)_to_(?P<dst>[0-9.]+T)$")
NIIGZ_RE = re.compile(r"\.nii\.gz$", re.IGNORECASE)


# Strict-validation error type prefixes (also embedded in submission_errors).
ERR_TASK_MISMATCH = "TASK_MISMATCH_ERROR"
ERR_FILE_TREE = "FILE_TREE_ERROR"
ERR_FORMAT = "FORMAT_ERROR"
ERR_SIZE = "SIZE_ERROR"
ERR_NAN = "NAN_ERROR"


class SubmissionValidationError(Exception):
    """Raised by validate_submission when participant data fails strict
    pre-checks. The message starts with one of the ERR_* prefixes
    (FILE_TREE_ERROR / FORMAT_ERROR / SIZE_ERROR / NAN_ERROR) so the
    platform can categorize failures by the prefix string."""
    pass


# Strict filename patterns. GT and the submission share the same
# convention: P_{MOD}_{FIELD}_{ID:04d}.nii.gz (and the _seg variant).
_PRED_NAME_RE = re.compile(
    r"^P_(?P<mod>T1W|T2W|T2FLAIR)_(?P<field>[0-9.]+T)_(?P<id>\d{4})\.nii\.gz$"
)
_SEG_NAME_RE = re.compile(
    r"^P_(?P<mod>T1W|T2W|T2FLAIR)_(?P<field>[0-9.]+T)_(?P<id>\d{4})_seg\.nii\.gz$"
)


# ============================================================
# SampleID
# ============================================================

@dataclass(frozen=True)
class SampleID:
    task: str
    pair: str
    modality: str
    subject_id: str          # canonical zero-padded form, e.g. "0001"
    src_field: str           # e.g. "0.1T"
    dst_field: str           # e.g. "7T"

    @property
    def subtask_key(self) -> str:
        return f"{self.pair}_{self.modality}"

    @property
    def gt_relpath(self) -> str:
        # GT layout (relative to gt_dir, which already points at task{N}/):
        # {modality}/{pair}/gt/P_{MOD}_{TGT}_{ID}.nii.gz
        return f"{self.modality}/{self.pair}/gt/P_{self.modality}_{self.dst_field}_{self.subject_id}.nii.gz"

    @property
    def gt_seg_relpath(self) -> str:
        return f"{self.modality}/{self.pair}/gt_seg/P_{self.modality}_{self.dst_field}_{self.subject_id}_seg.nii.gz"

    @property
    def pred_dir_relpath(self) -> str:
        # Submission layout (relative to input_root, which already points at task{N}/):
        # {modality}/{pair}/{pred|seg}/<file>.nii.gz
        return f"{self.modality}/{self.pair}/pred"

    @property
    def seg_dir_relpath(self) -> str:
        return f"{self.modality}/{self.pair}/seg"


# ============================================================
# Path / ID helpers
# ============================================================

def parse_pair(pair: str) -> Tuple[str, str]:
    m = PAIR_RE.match(pair)
    if not m:
        raise ValueError(f"Bad pair format: {pair!r} (expected '<src>T_to_<dst>T')")
    return m.group("src"), m.group("dst")


def canonical_id(raw: str) -> str:
    """Normalize subject ID strings ('1', '01', '0001', etc.) to a canonical 4-digit form."""
    digits = re.sub(r"[^0-9]", "", raw)
    if not digits:
        return raw
    return f"{int(digits):04d}"


def extract_id_from_filename(filename: str) -> Optional[str]:
    """Pull the numeric ID from the trailing token of a NIfTI filename.

    Submissions and GT now share the same naming convention:
    'P_{MOD}_{FIELD}_{ID:04d}.nii.gz' (with optional '_seg' suffix). Tolerant
    of legacy short forms where the ID is simply the trailing numeric token.
    """
    base = NIIGZ_RE.sub("", filename)
    # Strip optional _seg suffix (segmentation files).
    if base.endswith("_seg"):
        base = base[: -len("_seg")]
    parts = base.split("_")
    if not parts:
        return None
    last = parts[-1]
    if not re.fullmatch(r"\d+", last):
        return None
    return canonical_id(last)


def index_dir_by_id(dir_path: Path, suffix: str = ".nii.gz") -> Dict[str, Path]:
    """Index files in a directory by canonical subject ID."""
    out: Dict[str, Path] = {}
    if not dir_path.is_dir():
        return out
    for f in sorted(dir_path.iterdir()):
        if not f.name.lower().endswith(suffix):
            continue
        sid = extract_id_from_filename(f.name)
        if sid is not None:
            out[sid] = f
    return out


# ============================================================
# NIfTI I/O & metrics (self-contained, no cross-script imports)
# ============================================================

def load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return img.get_fdata(dtype=np.float32)


def get_voxel_volume(path: Path) -> float:
    import nibabel as nib
    img = nib.load(str(path))
    z = img.header.get_zooms()[:3]
    return float(np.prod(z))


def compute_nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    norm = np.linalg.norm(target)
    if norm < 1e-10:
        return 0.0
    return float(np.linalg.norm(pred - target) / norm)


def compute_ssim(pred: np.ndarray, target: np.ndarray, slice_axis: int = 2) -> float:
    from skimage.metrics import structural_similarity
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    data_range = target.max() - target.min()
    if data_range < 1e-10:
        return 1.0
    if pred.ndim == 2:
        return float(structural_similarity(pred, target, data_range=data_range))
    vals = []
    for i in range(pred.shape[slice_axis]):
        idx = [slice(None)] * pred.ndim
        idx[slice_axis] = i
        idx = tuple(idx)
        ps, ts = pred[idx], target[idx]
        if ts.max() - ts.min() < 1e-10:
            continue
        vals.append(structural_similarity(ps, ts, data_range=data_range))
    return float(np.mean(vals)) if vals else 1.0


# Lazy-init LPIPS: load model once and reuse.
_LPIPS_FN = None
_LPIPS_DEVICE = None


def _get_lpips(device: str):
    global _LPIPS_FN, _LPIPS_DEVICE
    if _LPIPS_FN is not None and _LPIPS_DEVICE == device:
        return _LPIPS_FN
    import torch
    import lpips as lpips_module
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    fn = lpips_module.LPIPS(net="alex").to(device)
    fn.eval()
    _LPIPS_FN = fn
    _LPIPS_DEVICE = device
    return fn


def compute_lpips(pred: np.ndarray, target: np.ndarray,
                  slice_axis: int = 2, device: str = "cuda") -> float:
    import torch
    fn = _get_lpips(device)
    real_device = _LPIPS_DEVICE
    pn = pred.astype(np.float64) * 2.0 - 1.0
    tn = target.astype(np.float64) * 2.0 - 1.0

    def _2d(p: np.ndarray, t: np.ndarray) -> float:
        pt = torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(real_device)
        tt = torch.from_numpy(t).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(real_device)
        with torch.no_grad():
            return float(fn(pt, tt).item())

    if pn.ndim == 2:
        return _2d(pn, tn)
    vals = []
    for i in range(pn.shape[slice_axis]):
        idx = [slice(None)] * pn.ndim
        idx[slice_axis] = i
        idx = tuple(idx)
        if np.abs(tn[idx]).max() < 1e-10:
            continue
        vals.append(_2d(pn[idx], tn[idx]))
    return float(np.mean(vals)) if vals else 0.0


def compute_dice_mean(seg_pred: np.ndarray, seg_target: np.ndarray) -> float:
    scores = []
    for lid in DGM_LABELS.values():
        pm = seg_pred == lid
        tm = seg_target == lid
        total = int(np.sum(pm)) + int(np.sum(tm))
        if total == 0:
            scores.append(1.0)
        else:
            inter = int(np.sum(pm & tm))
            scores.append(2.0 * inter / total)
    return float(np.mean(scores))


def compute_volume_mean(seg_pred: np.ndarray, seg_target: np.ndarray,
                        voxel_volume: float = 1.0) -> float:
    scores = []
    for lid in DGM_LABELS.values():
        vp = float(np.sum(seg_pred == lid)) * voxel_volume
        vt = float(np.sum(seg_target == lid)) * voxel_volume
        if vt < 1e-10:
            scores.append(1.0 if vp < 1e-10 else 0.0)
        else:
            scores.append(1.0 - abs(vp - vt) / vt)
    return float(np.mean(scores))


# ============================================================
# Output writers
# ============================================================

def write_invalid_results(output_dir: Path, error_message: str) -> Path:
    """Drop a minimal results.json with submission_status=INVALID so the
    platform always has a JSON to read, even when scoring blows up."""
    result_root = output_dir / "Result"
    result_root.mkdir(parents=True, exist_ok=True)
    summary = OrderedDict([
        ("submission_status", "INVALID"),
        ("submission_errors", error_message[:500]),
    ])
    json_path = result_root / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return json_path


def package_result_log(output_dir: Path) -> Path:
    """Bundle the entire Result/ directory into output_dir/better_log.zip
    (parity with the reference scorer). Internal arcname is `Result/...`,
    so unzipping reproduces the same layout."""
    result_root = output_dir / "Result"
    zip_path = output_dir / "better_log.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(result_root.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=f"Result/{f.relative_to(result_root)}")
    return zip_path


# ============================================================
# Sample enumeration & evaluation
# ============================================================

def enumerate_expected_samples(task: str, gt_dir: Path) -> List[SampleID]:
    """Enumerate expected (pair, modality, subject) samples by walking GT.

    `gt_dir` already points at the per-task GT root (its last path
    component is `task{N}`), so the layout is:
      {gt_dir}/{modality}/{pair}/gt/P_{MOD}_{TGT}_{ID}.nii.gz
    Expected = every subject that has GT under that pair's `gt/` directory,
    so the GT tree itself decides the subject set per (modality, pair).
    """
    cfg = TASK_CONFIG[task]
    out: List[SampleID] = []
    for pair in cfg["pairs"]:
        src, dst = parse_pair(pair)
        for modality in MODALITIES:
            gt_subdir = gt_dir / modality / pair / "gt"
            id_to_path = index_dir_by_id(gt_subdir)
            for sid in sorted(id_to_path.keys()):
                out.append(SampleID(
                    task=task, pair=pair, modality=modality,
                    subject_id=sid, src_field=src, dst_field=dst,
                ))
    return out


# ============================================================
# Submission validation (strict, fail-fast)
#
# Any participant-side defect (extra/missing case, wrong filename, bad
# shape, NaN) raises SubmissionValidationError; main() then writes
# results.json with submission_status=INVALID and the prefixed error
# message in submission_errors. No metric is computed for a submission
# that fails any of these checks.
# ============================================================

def _walk_files(root: Path):
    """Iterate every regular file under root, skipping macOS metadata."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name == ".DS_Store" or name.startswith("._"):
            continue
        if "__MACOSX" in p.parts:
            continue
        yield p


def _check_filename_formats(input_root: Path, needs_seg: bool) -> None:
    """Every regular file under input_root must sit at
    `{modality}/{pair}/{pred|seg}/<filename>` and follow the strict
    naming convention. Raises FORMAT_ERROR otherwise."""
    for f in _walk_files(input_root):
        rel = f.relative_to(input_root)
        parts = rel.parts
        if len(parts) != 4:
            raise SubmissionValidationError(
                f"{ERR_FORMAT}: unexpected file location: {rel.as_posix()} "
                f"(expected {{modality}}/{{pair}}/{{pred|seg}}/<filename>.nii.gz)"
            )
        modality, pair, kind, name = parts
        if modality not in MODALITIES:
            raise SubmissionValidationError(
                f"{ERR_FORMAT}: unexpected modality directory: {modality!r}"
            )
        if not PAIR_RE.match(pair):
            raise SubmissionValidationError(
                f"{ERR_FORMAT}: bad pair directory: {pair!r}"
            )
        if kind == "pred":
            if not _PRED_NAME_RE.fullmatch(name):
                raise SubmissionValidationError(
                    f"{ERR_FORMAT}: bad pred filename: {rel.as_posix()} "
                    f"(expected P_{{MOD}}_{{FIELD}}_{{ID:04d}}.nii.gz)"
                )
        elif kind == "seg":
            if not needs_seg:
                raise SubmissionValidationError(
                    f"{ERR_FORMAT}: seg/ files not allowed for this task: "
                    f"{rel.as_posix()}"
                )
            if not _SEG_NAME_RE.fullmatch(name):
                raise SubmissionValidationError(
                    f"{ERR_FORMAT}: bad seg filename: {rel.as_posix()} "
                    f"(expected P_{{MOD}}_{{FIELD}}_{{ID:04d}}_seg.nii.gz)"
                )
        else:
            raise SubmissionValidationError(
                f"{ERR_FORMAT}: unexpected directory under {modality}/{pair}/: "
                f"{kind!r} (expected pred or seg)"
            )


def _any_seg_files(input_root: Path,
                   samples_by_subtask: Dict[Tuple[str, str], List[SampleID]]) -> bool:
    """Detect whether the submission has at least one seg file anywhere."""
    for (modality, pair) in samples_by_subtask:
        seg_dir = input_root / modality / pair / "seg"
        if not seg_dir.is_dir():
            continue
        for f in seg_dir.iterdir():
            if f.is_file() and f.name.lower().endswith("_seg.nii.gz"):
                return True
    return False


def _check_file_tree(
    input_root: Path,
    samples_by_subtask: Dict[Tuple[str, str], List[SampleID]],
    needs_seg: bool,
    seg_submitted: bool,
) -> None:
    """For every (modality, pair) subtask, the pred ID set must equal the
    GT ID set. If seg_submitted, the seg ID set must also equal the GT
    ID set. Raises FILE_TREE_ERROR on any mismatch."""
    for (modality, pair), sample_list in samples_by_subtask.items():
        gt_ids = {sid.subject_id for sid in sample_list}

        pred_dir = input_root / modality / pair / "pred"
        pred_ids = set(index_dir_by_id(pred_dir).keys())
        missing = sorted(gt_ids - pred_ids)
        extra = sorted(pred_ids - gt_ids)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if extra:
                parts.append(f"extra={extra}")
            raise SubmissionValidationError(
                f"{ERR_FILE_TREE}: pred {modality}/{pair}: " + ", ".join(parts)
            )

        if needs_seg and seg_submitted:
            seg_dir = input_root / modality / pair / "seg"
            seg_ids = set(index_dir_by_id(seg_dir).keys())
            missing_s = sorted(gt_ids - seg_ids)
            extra_s = sorted(seg_ids - gt_ids)
            if missing_s or extra_s:
                parts = []
                if missing_s:
                    parts.append(f"missing={missing_s}")
                if extra_s:
                    parts.append(f"extra={extra_s}")
                raise SubmissionValidationError(
                    f"{ERR_FILE_TREE}: seg {modality}/{pair}: " + ", ".join(parts)
                )


def _check_data(
    input_root: Path,
    gt_dir: Path,
    samples: List[SampleID],
    seg_submitted: bool,
) -> None:
    """Load every pred (and seg if submitted) once, verify shape against
    the corresponding GT, and check for NaN. Raises FORMAT_ERROR (load
    fails), SIZE_ERROR (shape mismatch), or NAN_ERROR. Evaluator-side
    GT load failures raise plain RuntimeError so main() reports them as
    INVALID with the original exception type rather than a participant
    error prefix."""
    for sid in samples:
        pred_name = f"P_{sid.modality}_{sid.dst_field}_{sid.subject_id}.nii.gz"
        pred_path = input_root / sid.pred_dir_relpath / pred_name
        gt_path = gt_dir / sid.gt_relpath
        try:
            pred_arr = load_nifti(pred_path)
        except Exception as e:
            raise SubmissionValidationError(
                f"{ERR_FORMAT}: cannot load {pred_name}: {type(e).__name__}: {e}"
            ) from None
        try:
            gt_arr = load_nifti(gt_path)
        except Exception as e:
            raise RuntimeError(
                f"cannot load GT {gt_path}: {type(e).__name__}: {e}"
            ) from None
        if pred_arr.shape != gt_arr.shape:
            raise SubmissionValidationError(
                f"{ERR_SIZE}: {pred_name}: pred{pred_arr.shape} vs gt{gt_arr.shape}"
            )
        if np.isnan(pred_arr).any():
            raise SubmissionValidationError(f"{ERR_NAN}: {pred_name} contains NaN")
        del pred_arr, gt_arr
        gc.collect()

    if seg_submitted:
        for sid in samples:
            seg_name = f"P_{sid.modality}_{sid.dst_field}_{sid.subject_id}_seg.nii.gz"
            seg_path = input_root / sid.seg_dir_relpath / seg_name
            try:
                seg_arr = load_nifti(seg_path)
            except Exception as e:
                raise SubmissionValidationError(
                    f"{ERR_FORMAT}: cannot load {seg_name}: {type(e).__name__}: {e}"
                ) from None
            gt_seg_path = gt_dir / sid.gt_seg_relpath
            if gt_seg_path.exists():
                try:
                    gt_seg_arr = load_nifti(gt_seg_path)
                except Exception as e:
                    raise RuntimeError(
                        f"cannot load gt_seg {gt_seg_path}: {type(e).__name__}: {e}"
                    ) from None
                if seg_arr.shape != gt_seg_arr.shape:
                    raise SubmissionValidationError(
                        f"{ERR_SIZE}: {seg_name}: "
                        f"seg{seg_arr.shape} vs gt_seg{gt_seg_arr.shape}"
                    )
                del gt_seg_arr
            if np.isnan(seg_arr).any():
                raise SubmissionValidationError(f"{ERR_NAN}: {seg_name} contains NaN")
            del seg_arr
            gc.collect()


def validate_submission(
    input_root: Path,
    gt_dir: Path,
    task: str,
    samples: List[SampleID],
) -> Dict[str, bool]:
    """Strict pre-validation. Order: filename formats → file tree →
    per-file load + shape + NaN. Any participant-side failure raises
    SubmissionValidationError with an ERR_* prefix; evaluator-side
    failures raise RuntimeError.

    Returns:
      {"seg_submitted": bool}
        True iff this is a task1/task2 submission with at least one
        seg file. False for task3, or for task1/task2 that omit seg
        entirely (in which case Dice/Volume are reported as null).
    """
    cfg = TASK_CONFIG[task]
    needs_seg = cfg["needs_seg"]

    samples_by_subtask: Dict[Tuple[str, str], List[SampleID]] = {}
    for sid in samples:
        samples_by_subtask.setdefault((sid.modality, sid.pair), []).append(sid)

    _check_filename_formats(input_root, needs_seg)

    seg_submitted = needs_seg and _any_seg_files(input_root, samples_by_subtask)
    _check_file_tree(input_root, samples_by_subtask, needs_seg, seg_submitted)

    _check_data(input_root, gt_dir, samples, seg_submitted)

    return {"seg_submitted": seg_submitted}


def evaluate_one_sample(
    sid: SampleID,
    gt_dir: Path,
    pred_path: Path,
    seg_pred_path: Optional[Path],
    seg_gt_path: Optional[Path],
    metrics: List[str],
    needs_seg: bool,
    seg_submitted: bool,
    device: str,
) -> Dict[str, object]:
    """Compute metrics for a single pre-validated sample.
    `validate_submission` must run first, so all paths are guaranteed
    to exist and load cleanly. `seg_submitted=False` (task1/task2 with
    no seg files) is the one non-error path that yields Dice/Volume = None.
    """
    rec: Dict[str, object] = {
        "Task": sid.task,
        "Pair": sid.pair,
        "Modality": sid.modality,
        "SubjectID": sid.subject_id,
        "SubtaskKey": sid.subtask_key,
        "PredFile": pred_path.name,
        "SegFile": "",
        "Comments": "",
    }

    pred_arr = load_nifti(pred_path)
    gt_arr = load_nifti(gt_dir / sid.gt_relpath)
    if "nRMSE" in metrics:
        rec["nRMSE"] = compute_nrmse(pred_arr, gt_arr)
    if "SSIM" in metrics:
        rec["SSIM"] = compute_ssim(pred_arr, gt_arr)
    if "LPIPS" in metrics:
        rec["LPIPS"] = compute_lpips(pred_arr, gt_arr, device=device)
    del pred_arr, gt_arr
    gc.collect()

    if needs_seg and ("Dice" in metrics or "Volume" in metrics):
        if not seg_submitted:
            if "Dice" in metrics:
                rec["Dice"] = None
            if "Volume" in metrics:
                rec["Volume"] = None
            rec["Comments"] = "seg not submitted (task-level)"
        else:
            rec["SegFile"] = seg_pred_path.name
            import nibabel as nib
            seg_pred = nib.load(str(seg_pred_path)).get_fdata().astype(np.int32)
            seg_gt = nib.load(str(seg_gt_path)).get_fdata().astype(np.int32)
            if "Dice" in metrics:
                rec["Dice"] = compute_dice_mean(seg_pred, seg_gt)
            if "Volume" in metrics:
                rec["Volume"] = compute_volume_mean(seg_pred, seg_gt, get_voxel_volume(seg_gt_path))
            del seg_pred, seg_gt
            gc.collect()

    return rec


# ============================================================
# Aggregation
# ============================================================

def build_summary(
    records: List[Dict[str, object]],
    task: str,
    expected_per_subtask: Dict[str, int],
    seg_submitted: bool,
) -> "OrderedDict[str, object]":
    """Aggregate per-sample records into the platform-facing summary.
    Strict pre-validation guarantees every record is complete (no
    missing pred, no NaN), so per-metric values are simple means.
    `seg_submitted=False` (task1/task2 with no seg) reports Dice/Volume
    slots as None. Always False for task3."""
    metrics = TASK_CONFIG[task]["metrics"]
    seg_metrics = {"Dice", "Volume"}

    summary: "OrderedDict[str, object]" = OrderedDict()
    summary["submission_status"] = "SCORED"
    summary["primary_metric"] = "SSIM"
    summary["primary_score"] = None  # filled at the end

    total_expected = sum(expected_per_subtask.values())
    summary["Num_Files"] = f"{len(records)}/{total_expected}"

    by_subtask: Dict[str, List[Dict[str, object]]] = {s: [] for s in expected_per_subtask}
    for r in records:
        by_subtask[str(r["SubtaskKey"])].append(r)

    def metric_mean(rs: List[Dict[str, object]], met: str) -> Optional[float]:
        if not seg_submitted and met in seg_metrics:
            return None
        return round(float(np.mean([r[met] for r in rs])), 6)

    for stk in sorted(expected_per_subtask):
        rs = by_subtask[stk]
        summary[f"{stk}_Num"] = f"{len(rs)}/{expected_per_subtask[stk]}"
        for met in metrics:
            summary[f"{stk}_{met}_adj"] = metric_mean(rs, met)

    for met in metrics:
        summary[f"Mean_of_all_subtasks_{met}_adj"] = metric_mean(records, met)

    summary["primary_score"] = summary["Mean_of_all_subtasks_SSIM_adj"]
    return summary


def build_summary_per_modality(
    records: List[Dict[str, object]],
    task: str,
    expected_per_modality: Dict[str, int],
    total_expected: int,
    seg_submitted: bool,
) -> "OrderedDict[str, object]":
    """JSON summary aggregated per modality (T1W / T2W / T2FLAIR). Used for
    all three tasks; metrics come from TASK_CONFIG[task].

    Layout:
      header (4):        submission_status, primary_metric, primary_score, Num_Files
      per-modality:      Avg_{MOD}_Num, Avg_{MOD}_{metric}_adj × |metrics|
      global:            Mean_of_all_subtasks_{metric}_adj × |metrics|

    Sizes:
      - task1 / task2: 4 + 3 × (1 + 5) + 5 = 27 keys (5 metrics: nRMSE, SSIM, LPIPS, Dice, Volume)
      - task3:         4 + 3 × (1 + 3) + 3 = 19 keys (3 metrics: nRMSE, SSIM, LPIPS)

    `seg_submitted=False` (task1/task2 with no seg) reports Dice/Volume slots as None.
    Always False for task3.
    """
    metrics = TASK_CONFIG[task]["metrics"]
    seg_metrics = {"Dice", "Volume"}
    summary: "OrderedDict[str, object]" = OrderedDict()
    summary["submission_status"] = "SCORED"
    summary["primary_metric"] = "SSIM"
    summary["primary_score"] = None
    summary["Num_Files"] = f"{len(records)}/{total_expected}"

    by_mod: Dict[str, List[Dict[str, object]]] = {m: [] for m in MODALITIES}
    for r in records:
        by_mod[str(r["Modality"])].append(r)

    def metric_mean(rs: List[Dict[str, object]], met: str) -> Optional[float]:
        if not seg_submitted and met in seg_metrics:
            return None
        if not rs:
            return None
        return round(float(np.mean([r[met] for r in rs])), 6)

    for mod in MODALITIES:
        rs = by_mod[mod]
        summary[f"Avg_{mod}_Num"] = f"{len(rs)}/{expected_per_modality.get(mod, 0)}"
        for met in metrics:
            summary[f"Avg_{mod}_{met}_adj"] = metric_mean(rs, met)

    for met in metrics:
        summary[f"Mean_of_all_subtasks_{met}_adj"] = metric_mean(records, met)

    summary["primary_score"] = summary["Mean_of_all_subtasks_SSIM_adj"]
    return summary


# ============================================================
# Main task runner
# ============================================================

def run_task(
    input_root: Path,
    gt_dir: Path,
    output_dir: Path,
    task: str,
    device: str,
) -> None:
    cfg = TASK_CONFIG[task]
    metrics = cfg["metrics"]
    needs_seg = cfg["needs_seg"]

    samples = enumerate_expected_samples(task, gt_dir)
    if not samples:
        raise RuntimeError(
            f"No expected samples for {task}; check --gt_dir layout "
            f"(expected {{gt_dir}}/{task}/{{modality}}/{{pair}}/gt/)"
        )

    expected_per_subtask: Dict[str, int] = {}
    expected_per_modality: Dict[str, int] = {}
    for s in samples:
        expected_per_subtask[s.subtask_key] = expected_per_subtask.get(s.subtask_key, 0) + 1
        expected_per_modality[s.modality] = expected_per_modality.get(s.modality, 0) + 1

    print(f"Task {task}: {len(samples)} expected samples across "
          f"{len(expected_per_subtask)} subtasks", flush=True)

    # Strict pre-validation. Any participant-side defect raises
    # SubmissionValidationError up to main(), which writes results.json
    # with submission_status=INVALID and returns non-zero. No metric is
    # computed for a submission that fails any of these checks.
    val_info = validate_submission(input_root, gt_dir, task, samples)
    seg_submitted = val_info["seg_submitted"]
    print(f"Pre-validation OK (seg_submitted={seg_submitted})", flush=True)

    # Pre-build directory indices once instead of re-walking per sample.
    # Submission pred / seg are cached on first access (lazily, since most
    # submissions only populate a subset of pairs/modalities).
    pred_index_cache: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    seg_index_cache: Dict[Tuple[str, str, str], Dict[str, Path]] = {}

    def get_pred_index(sid: SampleID) -> Dict[str, Path]:
        key = (sid.task, sid.pair, sid.modality)
        if key not in pred_index_cache:
            pred_index_cache[key] = index_dir_by_id(input_root / sid.pred_dir_relpath)
        return pred_index_cache[key]

    def get_seg_index(sid: SampleID) -> Dict[str, Path]:
        key = (sid.task, sid.pair, sid.modality)
        if key not in seg_index_cache:
            seg_index_cache[key] = index_dir_by_id(input_root / sid.seg_dir_relpath)
        return seg_index_cache[key]

    records: List[Dict[str, object]] = []
    t0 = time.time()
    for i, sid in enumerate(samples, 1):
        pred_path = get_pred_index(sid).get(sid.subject_id)
        seg_pred_path: Optional[Path] = None
        seg_gt_path: Optional[Path] = None
        if needs_seg:
            seg_pred_path = get_seg_index(sid).get(sid.subject_id)
            gt_seg_candidate = gt_dir / sid.gt_seg_relpath
            if gt_seg_candidate.exists():
                seg_gt_path = gt_seg_candidate

        rec = evaluate_one_sample(
            sid=sid,
            gt_dir=gt_dir,
            pred_path=pred_path,
            seg_pred_path=seg_pred_path,
            seg_gt_path=seg_gt_path,
            metrics=metrics,
            needs_seg=needs_seg,
            seg_submitted=seg_submitted,
            device=device,
        )
        records.append(rec)
        if i % 10 == 0 or i == len(samples):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(samples)}] {sid.subtask_key}/{sid.subject_id} "
                  f"({elapsed:.1f}s)", flush=True)

    summary = build_summary(records, task, expected_per_subtask, seg_submitted)

    # JSON ships a per-modality aggregation for all three tasks (T1W / T2W /
    # T2FLAIR). xlsx still uses `summary`, so result_{task}.xlsx's summary
    # sheet preserves the per-(pair, modality) breakdown for inspection.
    summary_json = build_summary_per_modality(
        records, task, expected_per_modality, sum(expected_per_subtask.values()), seg_submitted
    )

    result_root = output_dir / "Result"
    result_root.mkdir(parents=True, exist_ok=True)

    json_path = result_root / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")

    # Optional Excel detail (skip silently if pandas/xlsxwriter unavailable).
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        excel_path = result_root / f"result_{task}.xlsx"
        summary_df = pd.DataFrame(
            [{"Category": k, "Value": v} for k, v in summary.items()]
        )
        with pd.ExcelWriter(str(excel_path), engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="results", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
        print(f"Wrote {excel_path}")
    except Exception as e:
        print(f"Skipped Excel output ({type(e).__name__}: {e})")

    zip_path = package_result_log(output_dir)
    print(f"Wrote {zip_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MRIxFields2026 — Validation Scorer"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Submission directory whose last path component is task{N} "
             "(contains {T1W,T2W,T2FLAIR}/{pair}/{pred,seg}/...). "
             "This scorer does not extract archives — unzip ahead of time.",
    )
    parser.add_argument(
        "-t", "--task", required=True,
        choices=list(TASK_CONFIG.keys()),
        help="Which task to score; cross-validates -i and -g (both must "
             "have this name as their last path component).",
    )
    parser.add_argument(
        "-g", "--gt_dir", required=True,
        help="Per-task GT directory whose last path component is task{N} "
             "(contains {T1W,T2W,T2FLAIR}/{pair}/{gt,gt_seg}/...).",
    )
    parser.add_argument(
        "-o", "--output", default="./",
        help="Output directory (Result/ will be created here)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device for LPIPS (default cuda; falls back to cpu)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gt_dir = Path(args.gt_dir).resolve()
        if not input_path.is_dir():
            raise SubmissionValidationError(
                f"{ERR_FORMAT}: --input is not a directory: {input_path} "
                f"(this scorer does not extract archives — unzip ahead of time)"
            )
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"--gt_dir not a directory: {gt_dir}")

        # Three-way cross-validation: -i, -g, and -t must agree on the task.
        if input_path.name != args.task:
            raise SubmissionValidationError(
                f"{ERR_TASK_MISMATCH}: --input last path component "
                f"{input_path.name!r} != --task {args.task!r}"
            )
        if gt_dir.name != args.task:
            raise RuntimeError(
                f"--gt_dir last path component {gt_dir.name!r} != "
                f"--task {args.task!r}"
            )

        run_task(
            input_root=input_path,
            gt_dir=gt_dir,
            output_dir=output_dir,
            task=args.task,
            device=args.device,
        )
        results_json = output_dir / "Result" / "results.json"
        if not results_json.exists():
            raise RuntimeError(f"Scoring finished but {results_json} is missing")
    except SubmissionValidationError as e:
        # Participant-side error: message is already prefixed (TASK_MISMATCH_ERROR /
        # FILE_TREE_ERROR / FORMAT_ERROR / SIZE_ERROR / NAN_ERROR). Write it
        # verbatim into submission_errors and exit non-zero without a traceback.
        invalid_path = write_invalid_results(output_dir, str(e))
        print(f"Wrote INVALID {invalid_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Evaluator-side error: include the exception type so operators can
        # tell what blew up. Re-raise so the original traceback hits stderr.
        invalid_path = write_invalid_results(output_dir, f"{type(e).__name__}: {e}")
        print(f"Wrote INVALID {invalid_path}: {type(e).__name__}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
