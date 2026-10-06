"""Base inference utilities for MRIxFields2026.

Contains shared logic: checkpoint loading, module-prefix stripping,
volume slice-by-slice prediction, and NIfTI I/O.
"""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from ..data.transforms import CenterCropOrPad
from ..data.utils import load_nifti, save_nifti


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by DDP wrapping."""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def extract_generator_state(state_dict: dict) -> dict:
    """Extract generator weights: prefer ema_state, fall back to generator key."""
    if state_dict.get("ema_state"):
        return strip_module_prefix(state_dict["ema_state"])
    if "generator" in state_dict:
        return strip_module_prefix(state_dict["generator"])
    # Fallback: extract netG from full model state dict
    if "model" in state_dict:
        model_sd = state_dict["model"]
        gen_sd = {k.replace("netG.", "", 1): v for k, v in model_sd.items()
                  if k.startswith("netG.")}
        if gen_sd:
            return strip_module_prefix(gen_sd)
    raise KeyError("No generator weights found (checked ema_state, generator, model.netG)")


def predict_volume_3d(
    model: torch.nn.Module | None,
    volume: np.ndarray,
    device: torch.device,
    mu_src: np.ndarray = None,
    sigma_src: np.ndarray = None,
    mu_tgt: np.ndarray = None,
    sigma_tgt: np.ndarray = None,
    force_norm: bool = False,
    roi_size: tuple | None = None,
    sw_overlap: float = 0.5,
) -> np.ndarray:
    """Run 3D inference on a full volume (C, D, H, W).

    When model is provided, uses MONAI SlidingWindowInferer to avoid OOM.
    When model is None, identity pass-through (baseline metrics).
    Normalises via mu_src/sigma_src and denormalises via mu_tgt/sigma_tgt.
    """
    try:
        from monai.inferers import SlidingWindowInferer
    except ImportError:
        SlidingWindowInferer = None

    C = volume.shape[0]
    has_norm = mu_src is not None

    if has_norm:
        def _to_list(x):
            return [x] if np.ndim(x) == 0 else list(x)
        mu_src = np.asarray(_to_list(mu_src), dtype=np.float32)
        sigma_src = np.asarray(_to_list(sigma_src), dtype=np.float32)
        mu_tgt = np.asarray(_to_list(mu_tgt), dtype=np.float32)
        sigma_tgt = np.asarray(_to_list(sigma_tgt), dtype=np.float32)

    # Per-channel preprocess
    proc = []
    for c in range(C):
        ch = volume[c].astype(np.float32).copy()
        ch = ch * 2.0 - 1.0
        _do_norm = has_norm and (model is not None or force_norm)
        if _do_norm:
            ch = (ch - mu_src[c]) / sigma_src[c]
        proc.append(ch)
    vol_norm = np.stack(proc, axis=0)  # (C, D, H, W)

    if model is None:
        pred_np = vol_norm
    else:
        if SlidingWindowInferer is None:
            raise ImportError("MONAI SlidingWindowInferer required for 3D model inference")
        rs = roi_size or tuple(min(d, 128) for d in volume.shape[-3:])
        # Ensure roi_size ≤ spatial dims
        spatial = volume.shape[-3:]
        rs = tuple(min(r, s) for r, s in zip(rs, spatial))
        inferer = SlidingWindowInferer(roi_size=rs, sw_batch_size=1, overlap=sw_overlap, mode="gaussian")
        tensor = torch.from_numpy(vol_norm).unsqueeze(0).float().to(device)  # (1, C, D, H, W)
        with torch.no_grad():
            out = inferer(tensor, model)
        pred_np = out[0].cpu().numpy()  # (C, D, H, W)

    _do_norm = has_norm and (model is not None or force_norm)
    if _do_norm:
        for c in range(C):
            pred_np[c] = pred_np[c] * sigma_tgt[c] + mu_tgt[c]
    else:
        pred_np = np.clip(pred_np, -1, 1) * 0.5 + 0.5

    # Brain mask (align spatial dims, then apply)
    brain_mask = (volume[0] > 0).astype(np.float32)
    for c in range(C):
        p, m = pred_np[c], brain_mask
        if p.shape != m.shape:
            msh = tuple(min(a, b) for a, b in zip(p.shape, m.shape))
            p, m = p[:msh[0], :msh[1], :msh[2]], m[:msh[0], :msh[1], :msh[2]]
        pred_np[c] = (p * m) if p.shape == m.shape else p
    return pred_np


def predict_volume_slice_by_slice(
    model: torch.nn.Module | None,
    volume: np.ndarray,
    device: torch.device,
    slice_axis: int = 2,
    crop_size: Tuple[int, int] = None,
    batch_size: int = 1,
    preprocess_fn=None,
    postprocess_fn=None,
    mu_src: np.ndarray = None,
    sigma_src: np.ndarray = None,
    mu_tgt: np.ndarray = None,
    sigma_tgt: np.ndarray = None,
    force_norm: bool = False,
) -> np.ndarray:
    """Run inference on a multi-channel volume, slice by slice.

    ``batch_size`` controls internal batching of slices from the SAME volume.

    Input:  volume (C, D, H, W)  — C=1 for single-modality, C=N for multi.
    Output: pred   (C, D, H, W)  — same shape.

    Each channel is normalized independently per-slice, then scaled to
    [-1,1] for the model, and mapped back to [0,1] in the output.

    When ``mu_src``/``sigma_src`` are provided, input is further normalised
    to N(0,1) and the model output is de-normalized by ``mu_tgt``/``sigma_tgt``.

    When ``model`` is ``None``, the pipeline acts as identity (preprocessed
    input is returned directly as the prediction).  This is used for computing
    baseline (source-vs-target) metrics.
    """
    C = volume.shape[0]
    # Account for channel dimension offset: for 4D (C, D, H, W) shift by 1
    slice_dim = slice_axis + (1 if volume.ndim == 4 else 0)
    n_slices = volume.shape[slice_dim]
    has_norm = mu_src is not None

    if has_norm:
        def _to_list(x):
            return [x] if np.ndim(x) == 0 else list(x)
        mu_src = np.asarray(_to_list(mu_src), dtype=np.float32)
        sigma_src = np.asarray(_to_list(sigma_src), dtype=np.float32)
        mu_tgt = np.asarray(_to_list(mu_tgt), dtype=np.float32)
        sigma_tgt = np.asarray(_to_list(sigma_tgt), dtype=np.float32)

    use_crop = crop_size is not None
    if use_crop:
        crop = CenterCropOrPad(crop_size)
        # Infer original spatial shape from a sample slice
        tmp_slicing = [slice(None)] * volume.ndim
        tmp_slicing[slice_dim] = 0
        tmp_slice = volume[tuple(tmp_slicing)]  # (C, H, W)
        orig_spatial = tmp_slice.shape[-2:]
        uncrop = CenterCropOrPad(orig_spatial)

    # Brain mask from first channel (shared anatomy)
    brain_mask = (volume[0] > 0).astype(np.float32)

    output = np.zeros_like(volume)  # (C, D, H, W)

    with torch.no_grad():
        batch_indices = []
        batch_tensors = []

        def _flush_batch():
            if not batch_indices:
                return
            tensor = torch.stack(batch_tensors, dim=0).to(device)  # [B, C, H, W]
            if preprocess_fn is not None:
                tensor = preprocess_fn(tensor)
            preds = tensor if model is None else model(tensor)  # [B, C, H, W]
            preds_np = preds.cpu().numpy()
            if postprocess_fn is not None:
                preds_np = postprocess_fn(preds_np)

            for idx, pred_np in zip(batch_indices, preds_np):
                if use_crop:
                    for c in range(C):
                        pred_np[c] = uncrop(pred_np[c])
                _do_norm = has_norm and (model is not None or force_norm)
                if _do_norm:
                    for c in range(C):
                        pred_np[c] = pred_np[c] * sigma_tgt[c] + mu_tgt[c]
                pred_np = np.clip(pred_np, -1, 1) * 0.5 + 0.5
                slicing = [slice(None)] * volume.ndim
                slicing[slice_dim] = idx
                output[tuple(slicing)] = pred_np

            batch_indices.clear()
            batch_tensors.clear()

        for i in range(n_slices):
            slicing = [slice(None)] * volume.ndim
            slicing[slice_dim] = i
            slc = volume[tuple(slicing)]  # (C, H, W)

            slc_input = np.stack([crop(slc[c]) for c in range(C)], axis=0) if use_crop else slc

            slc_scaled = slc_input * 2.0 - 1.0
            _do_norm = has_norm and (model is not None or force_norm)
            if _do_norm:
                for c in range(C):
                    slc_scaled[c] = (slc_scaled[c] - mu_src[c]) / sigma_src[c]
            tensor = torch.from_numpy(slc_scaled).float()  # (C, H, W)

            batch_indices.append(i)
            batch_tensors.append(tensor)

            if len(batch_indices) >= batch_size:
                _flush_batch()

        _flush_batch()

    for c in range(C):
        output[c] *= brain_mask
    return output


class BaseInference:
    """Base class for model-specific inference.

    Subclasses must implement ``load_generator()``.
    """

    def __init__(self, cfg: dict, checkpoint_path: str, device: torch.device,
                 source_field: str = None):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.modalities = self.cfg["data"].get("modalities", ["T1W"])
        self.n_channels = len(self.modalities)
        self._source_field = source_field
        self.model = self.load_generator()
        self.model.to(device).eval()

        # Normalisation stats (optional)
        import yaml as _yaml

        data_cfg = cfg.get("data", {})
        self._has_norm = data_cfg.get("normalize", False) or data_cfg.get("norm_nonzero", False)
        self._mu_src = self._sigma_src = self._mu_tgt = self._sigma_tgt = None
        if self._has_norm:
            if self._source_field is None:
                raise ValueError(
                    "source_field is required when normalize=true. "
                    "Pass --source_field for multi-field models."
                )
            stats_path = Path(self.checkpoint_path).parent.parent / "norm_stats.yaml"
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"norm_stats.yaml not found at {stats_path}. "
                    "This file is required when normalize=true."
                )
            ns_data = _yaml.safe_load(stats_path.read_text()) or {}
            tgt_field = data_cfg.get("target_field", "")
            src_norm = [ns_data[f"{m}/{self._source_field}"] for m in self.modalities]
            tgt_norm = [ns_data[f"{m}/{tgt_field}"] for m in self.modalities]
            self._mu_src = np.array([s["mu"] for s in src_norm], dtype=np.float32)
            self._sigma_src = np.array([s["sigma"] for s in src_norm], dtype=np.float32)
            self._mu_tgt = np.array([s["mu"] for s in tgt_norm], dtype=np.float32)
            self._sigma_tgt = np.array([s["sigma"] for s in tgt_norm], dtype=np.float32)

    def load_generator(self) -> torch.nn.Module:
        raise NotImplementedError

    def predict(
        self,
        volume: np.ndarray,
        slice_axis: int = 2,
        crop_size: Tuple[int, int] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        """Run inference on a volume.

        Accepts 3D (D, H, W) or 4D (C, D, H, W) input.
        Returns same ndim as input — 3D in → 3D out, 4D in → 4D out.

        When ``normalize: true`` in config, input is normalised by source
        μ/σ and output is de-normalised by target μ/σ.
        """
        was_3d = volume.ndim == 3
        if was_3d:
            volume = volume[np.newaxis, ...]
        out = predict_volume_slice_by_slice(
            self.model, volume, self.device,
            slice_axis=slice_axis, crop_size=crop_size, batch_size=batch_size,
            mu_src=self._mu_src, sigma_src=self._sigma_src,
            mu_tgt=self._mu_tgt, sigma_tgt=self._sigma_tgt,
        )
        return out[0] if was_3d else out

    def run_on_directory(
        self,
        input_dir,
        output_dir: Path,
        slice_axis: int = 2,
        crop_size: Tuple[int, int] = None,
        batch_size: int = 1,
    ):
        """Run inference. ``input_dir`` is a path or list of paths (one per modality).

        Single-modality -- ``input_dir=path`` or ``input_dir=[path]``:
            iterates over files in that path, saves directly to ``output_dir/``.

        Multi-modality -- ``input_dir=[path1, path2, ...]``:
            files matched by subject ID via ``extract_subject_id``.
            Saves per-modality results to ``output_dir/{modality}/``.
        """
        from ..data.utils import extract_subject_id

        if isinstance(input_dir, (str, Path)):
            input_dirs = [Path(input_dir)]
        else:
            input_dirs = [Path(d) for d in input_dir]

        C = len(input_dirs)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if C == 1:
            # Single-modality: iterate files directly
            nifti_files = sorted(input_dirs[0].rglob("*.nii.gz"))
            if not nifti_files:
                print(f"No .nii.gz files found in {input_dirs[0]}")
                return

            for base_path in tqdm(nifti_files, desc="Inference"):
                original_img = nib.load(str(base_path))
                original_affine = original_img.affine
                original_ornt = nib.io_orientation(original_affine)

                data, canonical_affine = load_nifti(base_path)
                canonical_ornt = nib.io_orientation(canonical_affine)

                pred = self.predict(data, slice_axis=slice_axis, crop_size=crop_size, batch_size=batch_size)

                transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
                pred = nib.orientations.apply_orientation(pred, transform)

                rel_path = base_path.relative_to(input_dirs[0])
                out_path = output_dir / rel_path
                save_nifti(pred, original_affine, out_path)

            print(f"Inference complete. Results saved to: {output_dir}")
            return

        # Multi-modality: match files by subject ID across directories
        lookups = []
        for i, d in enumerate(input_dirs):
            files = {extract_subject_id(f.name): f for f in d.rglob("*.nii.gz")}
            lookups.append(files)

        # Find subjects common to all modalities
        common_ids = set(lookups[0].keys())
        for lk in lookups[1:]:
            common_ids &= set(lk.keys())

        if not common_ids:
            print("No subjects found across all input directories.")
            return

        for subj_id in tqdm(sorted(common_ids), desc="Inference"):
            channels = []
            for i, d in enumerate(input_dirs):
                mod_path = lookups[i][subj_id]
                data, canonical_affine = load_nifti(mod_path)
                channels.append(data)

            volume = np.stack(channels, axis=0)  # (C, D, H, W)
            pred = self.predict(volume, slice_axis=slice_axis, crop_size=crop_size, batch_size=batch_size)

            for c in range(C):
                ch = pred[c]
                canonical_ornt = nib.io_orientation(canonical_affine)
                original_ornt = nib.io_orientation(nib.load(str(lookups[0][subj_id])).affine)
                transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
                ch = nib.orientations.apply_orientation(ch, transform)

                mod_out = output_dir / self.modalities[c]
                mod_out.mkdir(parents=True, exist_ok=True)
                out_path = mod_out / lookups[c][subj_id].name
                save_nifti(ch, nib.load(str(lookups[0][subj_id])).affine, out_path)

        print(f"Inference complete. Results saved to: {output_dir}")

__all__ = [
    "BaseInference",
    "strip_module_prefix",
    "extract_generator_state",
    "predict_volume_slice_by_slice",
    "predict_volume_3d",
]
