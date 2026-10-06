"""Torch-based metric computation for validation during training.

Replaces numpy-based metrics from ``Evaluation/evaluate.py`` with pure torch
operations so all computation stays on the same device as the model.
"""

import torch
import torch.nn.functional as F
import numpy as np


def _gaussian_kernel_2d(channels: int, kernel_size: int, sigma: float, device):
    """Return a ``(channels, 1, kernel_size, kernel_size)`` Gaussian kernel."""
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]         # 2D outer product
    return kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = None) -> float:
    """Peak Signal-to-Noise Ratio.

    Args:
        pred, target: ``(D, H, W)`` or ``(H, W)`` torch tensors.
        data_range: if ``None``, uses ``target.max() - target.min()``.

    Returns:
        Scalar PSNR in dB.
    """
    mse = (pred - target).pow(2).mean()
    if mse < 1e-10:
        return float("inf")
    dr = data_range if data_range is not None else (target.max() - target.min())
    if dr < 1e-10:
        return float("inf")
    return float(20.0 * torch.log10(dr / mse.sqrt()).item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 data_range: float = 2.0,
                 kernel_size: int = 11, sigma: float = 1.5,
                 k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (3D volume → slice-wise average).

    Implementation follows the standard SSIM definition using a Gaussian
    filter via separable ``conv2d``.

    Args:
        pred, target: ``(D, H, W)`` or ``(H, W)`` tensors.
        data_range: dynamic range of the input (default 2.0 for *2-1 space).

    Returns:
        Mean SSIM across all slices.
    """
    is_2d = pred.ndim == 2
    if is_2d:
        return _ssim_2d(pred, target, data_range, kernel_size, sigma, k1, k2)
    # 3D: average over the slice dimension
    vals = []
    for i in range(pred.shape[0]):
        vals.append(_ssim_2d(pred[i], target[i], data_range, kernel_size, sigma, k1, k2))
    return float(np.mean(vals)) if vals else 1.0


def _ssim_2d(pred, target, data_range, kernel_size, sigma, k1, k2):
    device = pred.device
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    p = pred.view(1, 1, *pred.shape)      # (1, 1, H, W)
    t = target.view(1, 1, *target.shape)

    kernel = _gaussian_kernel_2d(1, kernel_size, sigma, device)
    pad = kernel_size // 2

    mu_x = F.conv2d(p, kernel, padding=pad)
    mu_y = F.conv2d(t, kernel, padding=pad)

    sigma_x = F.conv2d(p ** 2, kernel, padding=pad) - mu_x ** 2
    sigma_y = F.conv2d(t ** 2, kernel, padding=pad) - mu_y ** 2
    sigma_xy = F.conv2d(p * t, kernel, padding=pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return float(ssim_map.mean().item())


