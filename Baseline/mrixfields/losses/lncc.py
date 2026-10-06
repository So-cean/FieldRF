"""Localized Normalized Cross-Correlation (LNCC) loss.

LNCC is invariant to intensity offset/scaling, making it suitable for
cross-field harmonization where intensity distributions differ significantly
(e.g. 0.1T vs 7T MRI).  It measures local shape similarity within a sliding
window, independent of absolute intensity.

Input range: any (LNCC internally zero-means).  Single-channel 2D only.

Engineering notes:
- Constant windows (e.g. uniform background) produce zero vectors after
  zero-mean.  F.cosine_similarity returns 0 for zero vectors, which would
  give loss=1 for matching backgrounds.  Fixed via torch.where mask.
- Boundary uses reflect padding to avoid zero-padding bias.
- Single-channel only (C=1).  Multi-channel would require per-channel LNCC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LNCCloss(nn.Module):
    """Localized Normalized Cross-Correlation loss.

    For each sliding window: zero-mean → cosine similarity (= Pearson r).
    Loss = 1 - mean(NCC).  Constant windows (both sides) treated as perfect match.

    Args:
        kernel_size: Sliding window size (default 9).
        eps: Norm threshold for detecting constant windows (default 1e-6).
    """

    def __init__(self, kernel_size: int = 9, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        p = F.pad(pred, [k // 2] * 4, mode='reflect')
        t = F.pad(target, [k // 2] * 4, mode='reflect')

        pd = F.unfold(p, k)
        td = F.unfold(t, k)

        pd = pd - pd.mean(dim=1, keepdim=True)
        td = td - td.mean(dim=1, keepdim=True)

        cc = F.cosine_similarity(pd, td, dim=1)

        # Only when BOTH sides are constant windows is NCC undefined
        both_const = (pd.norm(dim=1) < self.eps) & (td.norm(dim=1) < self.eps)
        cc = torch.where(both_const, torch.ones_like(cc), cc)

        return 1.0 - cc.mean()
