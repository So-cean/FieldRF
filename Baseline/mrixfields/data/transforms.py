"""Preprocessing transforms for MRI volumes and 2D slices."""

from typing import Optional, Tuple

import numpy as np
import torch


class CenterCropOrPad:
    """Center-crop or zero-pad a 3D volume or 2D slice to a target shape."""

    def __init__(self, target_shape: Tuple[int, ...]):
        self.target_shape = target_shape

    def __call__(self, data: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            dtype = data.dtype
            arr = data.cpu().numpy()
        else:
            dtype = data.dtype
            arr = data
        # Auto-pad target_shape for channel dim
        if arr.ndim > len(self.target_shape):
            target_shape = (arr.shape[0],) + self.target_shape
        else:
            target_shape = self.target_shape
        result = np.zeros(target_shape, dtype=np.float32)
        slices_src = []
        slices_dst = []
        for i, (s, t) in enumerate(zip(arr.shape, target_shape)):
            if s > t:
                start = (s - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(None))
            else:
                start = (t - s) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + s))
        result[tuple(slices_dst)] = arr[tuple(slices_src)]
        if is_tensor:
            return torch.from_numpy(result).to(dtype)
        return result


class NormalizeMinMax:
    """Min-max normalize a volume to [0, 1], with optional percentile clipping.

    If `clip_percentile` is set (e.g. 99.5), values are first clipped to that
    percentile then min-max rescaled. If `clip_percentile` is None (default),
    the volume is returned as float32 unchanged.
    """

    def __init__(self, clip_percentile: Optional[float] = None):
        self.clip_percentile = clip_percentile

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        if self.clip_percentile is None:
            return data
        upper = np.percentile(data, self.clip_percentile)
        data = np.clip(data, 0, upper)
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
        return data


class ToTensor:
    """Convert numpy array to PyTorch tensor, adding channel dimension."""

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data.copy()).unsqueeze(0).float()


class ScaleToMinusOneOne:
    """Scale tensor from [0, 1] to [-1, 1]: x * 2 - 1.

    Standard range for GAN training (CycleGAN, CUT, StarGAN v2).
    Matches Tanh generator output range.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2.0 - 1.0


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: np.ndarray) -> np.ndarray | torch.Tensor:
        for t in self.transforms:
            data = t(data)
        return data
