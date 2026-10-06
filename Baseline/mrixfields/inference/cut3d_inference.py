"""CUT3D inference (ResnetGenerator3D + SlidingWindowInferer).

Uses self-contained checkpoint format with _metadata.config_actual.
"""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from .base_inference import BaseInference, extract_generator_state
from ..data.utils import load_nifti, save_nifti
from ..models.networks_cut3d import ResnetGenerator3D

try:
    from monai.inferers import SlidingWindowInferer
    _has_monai = True
except ImportError:
    _has_monai = False


class CUT3DInference(BaseInference):
    """3D CUT inference using MONAI SlidingWindowInferer."""

    def __init__(self, cfg, checkpoint_path, device,
                 source_field=None, **kwargs):
        super().__init__(cfg, checkpoint_path, device, source_field=source_field)

    def load_generator(self):
        state = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False,
        )
        if "generator_config" in state and state["generator_config"].get("kwargs"):
            kw = state["generator_config"]["kwargs"].copy()
        else:
            kw = self.cfg.get("model", {}).copy()
            kw.setdefault("use_tanh", not self.cfg.get("data", {}).get("normalize", False))

        model = ResnetGenerator3D(
            input_nc=kw.get("input_nc", 1),
            output_nc=kw.get("output_nc", 1),
            ngf=kw.get("ngf", 32),
            n_blocks=kw.get("n_blocks", 9),
            use_tanh=kw.get("use_tanh", True),
            skip_dropout=kw.get("skip_dropout", 0.0),
        )
        gen_state = extract_generator_state(state)
        model.load_state_dict(gen_state)
        print(f"Loaded CUT3D generator from: {self.checkpoint_path}")
        return model

    def predict(
        self,
        volume: np.ndarray,
        slice_axis: int = 2,
        crop_size: Tuple[int, int] = None,
        batch_size: int = 1,
        roi_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
    ) -> np.ndarray:
        """Run 3D sliding-window inference on a volume.

        Accepts 3D (D, H, W) or 4D (C, D, H, W) input.
        Returns same ndim as input.
        """
        if not _has_monai:
            raise ImportError("MONAI is required for 3D sliding-window inference.")

        was_3d = volume.ndim == 3
        if was_3d:
            volume = volume[np.newaxis, ...]  # (1, D, H, W)

        C, D, H, W = volume.shape
        output = np.zeros_like(volume)

        brain_mask = (volume[0] > 0).astype(np.float32)

        for c in range(C):
            vol = volume[c].astype(np.float32)  # (D, H, W)

            # Per-slice percentile clip + scale to [-1, 1]
            upper = float(np.percentile(vol, 99.5))
            vol = np.clip(vol, 0, upper)
            if upper > 1e-8:
                vol = vol / upper
            vol = vol * 2.0 - 1.0

            # z-score normalisation (if configured)
            if self._has_norm:
                vol = (vol - self._mu_src[c]) / self._sigma_src[c]

            tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(self.device)
            # shape: (1, 1, D, H, W)

            inferer = SlidingWindowInferer(
                roi_size=roi_size,
                sw_batch_size=batch_size,
                overlap=overlap,
                mode="gaussian",
            )
            with torch.no_grad():
                pred = inferer(inputs=tensor, network=self.model)

            pred_np = pred.squeeze().cpu().numpy()

            # Denormalise if using z-score
            if self._has_norm:
                pred_np = pred_np * self._sigma_tgt[c] + self._mu_tgt[c]

            pred_np = np.clip(pred_np, -1, 1) * 0.5 + 0.5  # [-1,1] -> [0,1]
            pred_np = pred_np * brain_mask

            output[c] = pred_np.astype(np.float32)

        return output[0] if was_3d else output

    def run_on_directory(
        self,
        input_dir,
        output_dir: Path,
        slice_axis: int = 2,
        crop_size: Tuple[int, int] = None,
        batch_size: int = 1,
    ):
        """Run 3D inference on a directory of NIfTI files."""
        if isinstance(input_dir, (str, Path)):
            input_dirs = [Path(input_dir)]
        else:
            input_dirs = [Path(d) for d in input_dir]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        nifti_files = sorted(input_dirs[0].rglob("*.nii.gz"))
        if not nifti_files:
            print(f"No .nii.gz files found in {input_dirs[0]}")
            return

        infer_cfg = self.cfg.get("inference", {})
        roi_size = tuple(infer_cfg.get("roi_size", [128, 128, 128]))
        overlap = infer_cfg.get("overlap", 0.5)
        sw_batch_size = infer_cfg.get("batch_size", 1)

        for nifti_path in tqdm(nifti_files, desc="3D Inference"):
            original_img = nib.load(str(nifti_path))
            original_affine = original_img.affine
            original_ornt = nib.io_orientation(original_affine)

            data, canonical_affine = load_nifti(nifti_path)
            canonical_ornt = nib.io_orientation(canonical_affine)

            pred = self.predict(
                data, slice_axis=slice_axis,
                roi_size=roi_size, overlap=overlap,
                batch_size=sw_batch_size,
            )

            transform = nib.orientations.ornt_transform(canonical_ornt, original_ornt)
            pred = nib.orientations.apply_orientation(pred, transform)

            rel_path = nifti_path.relative_to(input_dirs[0])
            out_path = output_dir / rel_path
            save_nifti(pred, original_affine, out_path)

        print(f"3D Inference complete. Results saved to: {output_dir}")


__all__ = [
    "CUT3DInference",
]
