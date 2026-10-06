"""[FieldRF] Shared visualization utilities for epoch-end eval.

Used by FieldRF training, inference and visualization entry points.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged, EnsureTyped,
)
from skimage.metrics import structural_similarity
from models.checkpoint import load_vae_checkpoint
from data.volume_dataset import list_nifti_files

TARGET_SIZE = (384, 384, 384)

# ── VAE cache ──
_VAE_CACHE = {}


def load_vae(ckpt_path: str, device: torch.device) -> AutoencoderKlMaisi:
    """Load MAISI VAE for decode, cached across vis epochs."""
    cache_key = (ckpt_path, str(device))
    if cache_key in _VAE_CACHE:
        return _VAE_CACHE[cache_key]

    vae = AutoencoderKlMaisi(
        spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
        num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
        norm_num_groups=32, norm_eps=1e-6,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
        use_checkpointing=False, use_convtranspose=False,
        norm_float16=True, num_splits=4, dim_split=1,
    ).to(device)
    load_vae_checkpoint(vae, ckpt_path, map_location=device)
    vae.eval()
    _VAE_CACHE[cache_key] = vae
    return vae


def decode_volume(vae, z, sw_cfg, amp_dtype):
    """Decode central image-space 64-slice volume (roi_size=[384,384,64]).

    Latent: 64/4 = 16 slices from center of 96 → (B,1,384,384,64).
    """
    latent_z_slices = sw_cfg["roi_size"][2] // 4   # 64→16
    total_z = z.shape[4]                             # 96
    start = total_z // 2 - latent_z_slices // 2
    z_crop = z[..., start : start + latent_z_slices]
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
            return vae.decode_stage_2_outputs(z_crop)


def compute_metrics(gt_np: np.ndarray, pred_np: np.ndarray) -> dict:
    """Brain-masked PSNR/MAE + per-slice 2D SSIM (matches Evaluation/evaluate.py)."""
    mask = gt_np > 0
    gm, pm = gt_np[mask], pred_np[mask]

    mse = np.mean((gm - pm) ** 2) if len(gm) > 0 else 0
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    mae = np.mean(np.abs(gm - pm)) if len(gm) > 0 else 0

    data_range = gt_np.max() - gt_np.min()
    slice_vals = []
    for i in range(gt_np.shape[2]):
        ps, ts = pred_np[..., i], gt_np[..., i]
        if ts.max() - ts.min() < 1e-10:
            continue
        slice_vals.append(structural_similarity(ps, ts, data_range=data_range))
    ssim_v = float(np.mean(slice_vals)) if slice_vals else 0.0

    return {"psnr": psnr, "ssim": ssim_v, "mae": mae}


def get_brain_mask(data_root, split, modality, target_field, filename, roi_z):
    """Load original NIfTI → brain mask (bg<=0) aligned with decode z-range.

    Returns: np.ndarray of shape (384, 384, roi_z), dtype=bool.
    """

    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        SpatialPadd(keys=["image"], spatial_size=TARGET_SIZE, method="symmetric"),
        CenterSpatialCropd(keys=["image"], roi_size=TARGET_SIZE),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0),
        EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
    ])

    files = list_nifti_files(data_root, split, modality, target_field)
    for f in files:
        if f"_P_{modality}_{target_field}_{filename}" in str(f) or str(f).endswith(f"{filename}.nii.gz"):
            item = transform({"image": str(f)})
            tgt_01 = torch.clamp((item["image"][0] + 1) / 2, 0, 1).numpy()
            z_start = tgt_01.shape[2] // 2 - roi_z // 2
            z_end = z_start + roi_z
            return tgt_01[:, :, z_start:z_end] > 0

    # Fallback: return all-ones mask (no masking)
    print(f"    [vis] WARNING: brain mask not found for {filename}, using no mask")
    return np.ones((384, 384, roi_z), dtype=bool)


def render_vis_grid(rows_data, title, save_path, epoch):
    """Render side-by-side PNG: Source | Harmonized | Target.

    Args:
        rows_data: list of dicts with keys 'src_np', 'harm_np', 'tgt_np',
                   'source_field', 'filename', optionally 'psnr', 'ssim'.
        title: suptitle string.
        save_path: Path to save PNG.
        epoch: int epoch number (for filename if save_path not absolute).
    """
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows_data):
        src = row["src_np"]
        harm = row["harm_np"]
        tgt = row["tgt_np"]
        sf = row["source_field"]
        z_mid = src.shape[2] // 2

        axes[i][0].imshow(src[:, :, z_mid], cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[i][0].set_title(f"Source ({sf})")
        axes[i][0].axis("off")

        tgt_field = row.get("target_field", "7T")
        title_str = f"Harmonized"
        if "psnr" in row:
            title_str += f" | PSNR={row['psnr']:.1f} | SSIM={row['ssim']:.3f}"
        axes[i][1].imshow(harm[:, :, z_mid], cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[i][1].set_title(title_str)
        axes[i][1].axis("off")

        axes[i][2].imshow(tgt[:, :, z_mid], cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[i][2].set_title(f"Target ({tgt_field})")
        axes[i][2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def init_metrics_csv(csv_path: Path, columns: list) -> csv.writer:
    """Open metrics CSV, write header if new. Returns writer + file handle.

    Uses append mode — rows accumulate across epochs within the same run.
    Call archive_metrics_csvs() at training start to avoid mixing runs.
    """
    # A killed validation can leave a zero-byte path behind before Python's
    # buffered header is flushed. Treat that file as new so a retry remains a
    # valid CSV instead of silently starting with a data row.
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow(columns)
    return writer, f


def archive_metrics_csvs(vis_dir: Path, patterns: list = None):
    """Rename existing metrics CSVs with timestamp before a new training run.

    Prevents mixing data from different runs in the same CSV file.
    Existing file vis_metrics_T1W.csv → vis_metrics_T1W_20260714_1230.csv
    """
    import shutil
    from datetime import datetime

    if patterns is None:
        patterns = ["vis_metrics_*.csv", "vae_val_metrics_*.csv"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = Path(vis_dir)
    if not vis_dir.is_dir():
        return

    archived = 0
    for pattern in patterns:
        for f in vis_dir.glob(pattern):
            if not f.is_file():
                continue
            stem = f.stem  # e.g. "vis_metrics_T1W"
            new_name = f"{stem}_{ts}{f.suffix}"
            new_path = f.parent / new_name
            shutil.move(str(f), str(new_path))
            archived += 1
    if archived > 0:
        print(f"  Archived {archived} old CSV(s) with timestamp {ts}", flush=True)


# ────────────────────────────────────────────────────────────────────── #
#  RF evaluation-aligned vis helpers (decode-centered, eval-slice crop,
#  torch metrics matching Evaluation/evaluate.py, original-image plot)
# ────────────────────────────────────────────────────────────────────── #

import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation


def decode_volume_centered(vae, z, sw_cfg, amp_dtype, z_center):
    """Decode a 64-slice image window centered at `z_center` (RAS 384-space).

    Centered decoding maps the 96³ latent grid to the 384³ image grid (4×).
    Returns (B, 1, 384, 384, 64) tensor in [-1, 1].
    """
    latent_z_slices = sw_cfg["roi_size"][2] // 4        # 64 -> 16 latent slices
    total_z = z.shape[4]                                 # 96
    img_start = z_center - latent_z_slices * 2           # image-space start
    # round to nearest valid latent position (NOT floor // — that causes up to
    # 3-slice spatial misalignment when z_center-32 is not divisible by 4)
    start = max(0, min(total_z - latent_z_slices, round(img_start / 4)))
    z_crop = z[..., start: start + latent_z_slices]
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
            return vae.decode_stage_2_outputs(z_crop)


def find_nifti_path(data_root, split, modality, field, subject_id):
    """Return absolute path to P_{modality}_{field}_{subject_id}.nii.gz, or None."""
    target_name = f"P_{modality}_{field}_{subject_id}.nii.gz"
    for f in list_nifti_files(data_root, split, modality, str(field)):
        if Path(f).name == target_name:
            return str(f)
    for f in list_nifti_files(data_root, split, modality, str(field)):
        if f.name.endswith(f"{subject_id}.nii.gz"):
            return str(f)
    return None


_ORIG_TRANSFORM = None
def _get_orig_transform():
    global _ORIG_TRANSFORM
    if _ORIG_TRANSFORM is None:
        _ORIG_TRANSFORM = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            SpatialPadd(keys=["image"], spatial_size=TARGET_SIZE, method="symmetric"),
            CenterSpatialCropd(keys=["image"], roi_size=TARGET_SIZE),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0),
            EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
        ])
    return _ORIG_TRANSFORM


def load_orig_ras_384(nifti_path):
    """Load original NIfTI -> RAS -> pad/crop 384^3 -> [0,1] numpy (384,384,384).

    Returns (vol_384_01, orig_shape, pad_before_z, crop_start_z) where the last
    two describe how the original z-axis was padded/cropped into 384 — needed to
    map the eval slice range [150,180) (in original z) into 384-space.
    """
    img = nib.as_closest_canonical(nib.load(str(nifti_path)))
    orig_shape = img.shape[:3]
    ras_data = img.get_fdata(dtype=np.float32)  # RAS, raw intensity

    # Replicate SpatialPad + CenterCrop on the z-axis only (to infer the offset).
    sz = orig_shape[2]
    if sz < 384:
        pad_before_z = (384 - sz) // 2
        crop_start_z = 0
    elif sz > 384:
        pad_before_z = 0
        crop_start_z = (sz - 384) // 2
    else:
        pad_before_z = 0
        crop_start_z = 0

    # Use the MONAI transform for the actual volume (handles all axes consistently).
    item = _get_orig_transform()({"image": str(nifti_path)})
    vol_01 = torch.clamp((item["image"][0] + 1) / 2, 0, 1).numpy()  # (384,384,384) [0,1]
    return vol_01, orig_shape, pad_before_z, crop_start_z


def eval_window_in_384(pad_before_z, crop_start_z, eval_range=(150, 180)):
    """Map eval axial slice range (in original z) into 384-space indices.

    FieldRF preprocess: SpatialPad (symmetric, for sz<384) then CenterSpatialCrop
    (for sz>384). So:
      - sz < 384: original z=i -> 384 z = i + pad_before_z
      - sz > 384: original z=i -> 384 z = i - crop_start_z
      - sz = 384: identity
    Returns (z0_384, z1_384) for the eval range.
    """
    z0, z1 = eval_range
    if pad_before_z > 0:
        return z0 + pad_before_z, z1 + pad_before_z
    elif crop_start_z > 0:
        return z0 - crop_start_z, z1 - crop_start_z
    return z0, z1


# ── torch-efficient metrics matching Evaluation/evaluate.py (no brain mask) ── #

def torch_nrmse(pred, target):
    """nRMSE = ||pred - target|| / ||target|| over the full volume. No mask."""
    p = torch.as_tensor(pred, dtype=torch.float64)
    t = torch.as_tensor(target, dtype=torch.float64)
    norm = torch.linalg.vector_norm(t.flatten())
    if norm.item() < 1e-10:
        return 0.0
    return float(torch.linalg.vector_norm((p - t).flatten()) / norm)


def torch_psnr_slices(pred, target, slice_axis=2):
    """Per-slice PSNR (20*log10(1/sqrt(MSE))), averaged. Matches evaluate.compute_psnr."""
    p = torch.as_tensor(np.asarray(pred), dtype=torch.float64)
    t = torch.as_tensor(np.asarray(target), dtype=torch.float64)
    p = p.movedim(slice_axis, 0)  # (N, H, W)
    t = t.movedim(slice_axis, 0)
    # Skip slices where target has no contrast.
    span = t.amax(dim=(-2, -1)) - t.amin(dim=(-2, -1))  # (N,)
    valid = span > 1e-10
    if not valid.any():
        return float("inf")
    pv, tv = p[valid], t[valid]
    mse = ((pv - tv) ** 2).mean(dim=(-2, -1)).clamp(min=1e-10)  # (M,)
    return float((20.0 * torch.log10(1.0 / torch.sqrt(mse))).mean())


def torch_ssim_slices(pred, target, slice_axis=2, win=7):
    """Per-slice 2D SSIM (uniform window, reflect pad), averaged.

    Matches evaluate.compute_ssim (skimage structural_similarity with default
    win_size=7, gaussian_weights=False) to within ~1e-3. Vectorized over slices
    via a single conv2d.
    """
    p = torch.as_tensor(np.asarray(pred), dtype=torch.float64)
    t = torch.as_tensor(np.asarray(target), dtype=torch.float64)
    p = p.movedim(slice_axis, 0)  # (N, H, W)
    t = t.movedim(slice_axis, 0)
    data_range = (t.max() - t.min()).item()
    if data_range < 1e-10:
        return 1.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    pad = win // 2

    def _uf(x):  # uniform mean filter, reflect padding (matches scipy default)
        xp = F.pad(x.unsqueeze(1), (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(xp, kernel_size=win, stride=1)[:, 0]

    span = t.amax(dim=(-2, -1)) - t.amin(dim=(-2, -1))
    valid = span > 1e-10
    if not valid.any():
        return 1.0
    pv, tv = p[valid], t[valid]
    mu_x = _uf(pv); mu_y = _uf(tv)
    sig_x2 = _uf(pv * pv) - mu_x * mu_x
    sig_y2 = _uf(tv * tv) - mu_y * mu_y
    sig_xy = _uf(pv * tv) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sig_x2 + sig_y2 + C2)
    ssim_map = num / den
    # skimage crops the win_size//2 border when averaging.
    return float(ssim_map[:, pad:-pad, pad:-pad].mean())


def compute_metrics_eval(pred, target, slice_axis=2):
    """Evaluation-aligned metrics (no brain mask). Returns {nrmse, psnr, ssim}."""
    return {
        "nrmse": torch_nrmse(pred, target),
        "psnr": torch_psnr_slices(pred, target, slice_axis=slice_axis),
        "ssim": torch_ssim_slices(pred, target, slice_axis=slice_axis),
    }


def render_evaluation_grid(rows_data, title, save_path, epoch,
                       dec_label="dec", orig_label="orig", vae_label="vae"):
    """RF vis grid: Orig Source | Harm | Orig Target | Decoded Target.

    Each row is one (source_field -> target) pair. Metrics shown in the Harm
    column title:
      vae_label   : decoded_target vs orig_target  (VAE reconstruction ceiling)
      dec_label   : harm_decoded vs tgt_decoded     (bridge quality, VAE cancels)
      orig_label  : harm_decoded vs orig_target     (full pipeline incl. VAE)

    Args:
        dec_label: label for metrics_decoded (default "dec").
        orig_label: label for metrics_orig (default "orig").
        vae_label: label for metrics_vae (default "vae").
    """
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for i, row in enumerate(rows_data):
        src = row["orig_src"]
        harm = row["harm"]
        tgt = row["orig_tgt"]
        tgt_dec = row.get("tgt_decoded")
        sf = row["source_field"]
        z_mid = src.shape[2] // 2
        vmin, vmax = 0.0, 1.0

        axes[i][0].imshow(src[:, :, z_mid], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[i][0].set_title(f"Orig Source ({sf})")
        axes[i][0].axis("off")

        m_dec = row.get("metrics_decoded", {})
        m_orig = row.get("metrics_orig", {})
        m_vae = row.get("metrics_vae", {})
        title_str = "Harm"
        if m_vae:
            title_str += (f"\n{vae_label}:  nRMSE={m_vae['nrmse']:.3f} SSIM={m_vae['ssim']:.3f} "
                          f"PSNR={m_vae['psnr']:.1f}")
        if m_orig:
            title_str += (f"\n{orig_label}: nRMSE={m_orig['nrmse']:.3f} SSIM={m_orig['ssim']:.3f} "
                          f"PSNR={m_orig['psnr']:.1f}")
        if m_dec:
            title_str += (f"\n{dec_label}:  nRMSE={m_dec['nrmse']:.3f} SSIM={m_dec['ssim']:.3f} "
                          f"PSNR={m_dec['psnr']:.1f}")
        axes[i][1].imshow(harm[:, :, z_mid], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[i][1].set_title(title_str, fontsize=9)
        axes[i][1].axis("off")

        tgt_field = row.get("target_field", "7T")
        axes[i][2].imshow(tgt[:, :, z_mid], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[i][2].set_title(f"Orig Target ({tgt_field})")
        axes[i][2].axis("off")

        if tgt_dec is not None:
            axes[i][3].imshow(tgt_dec[:, :, z_mid], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            axes[i][3].set_title(f"Decoded Target (VAE recon)")
            axes[i][3].axis("off")
        else:
            axes[i][3].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def render_vis_grid_vae(rows_data, title, save_path, epoch):
    """VAE validation grid: 2 columns (Orig | Decoded).

    Each row is one (modality, field) pair. Metrics shown in the Decoded
    column title: nRMSE, SSIM, PSNR.

    Args:
        rows_data: list of dicts with keys 'orig', 'decoded', 'field',
                   'subject', 'metrics' (dict with nrmse/ssim/psnr).
    """
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for i, row in enumerate(rows_data):
        orig = row["orig"]
        decoded = row["decoded"]
        field = row["field"]
        subj = row["subject"]
        z_mid = orig.shape[2] // 2
        vmin, vmax = 0.0, 1.0

        axes[i][0].imshow(orig[:, :, z_mid], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[i][0].set_title(f"Orig ({field} {subj})")
        axes[i][0].axis("off")

        m = row.get("metrics", {})
        title_str = "Decoded"
        if m:
            title_str += (f"\nnRMSE={m['nrmse']:.3f} SSIM={m['ssim']:.3f} "
                          f"PSNR={m['psnr']:.1f}")
        axes[i][1].imshow(decoded[:, :, z_mid], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        axes[i][1].set_title(title_str, fontsize=9)
        axes[i][1].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
