"""MS-SSIM loss wrapper using pytorch_msssim."""

import torch.nn as nn
from pytorch_msssim import MS_SSIM


class MSSSIMLoss(nn.Module):
    """MS-SSIM loss: returns 1 - MS_SSIM(x, y) as a scalar.

    Delegates to pytorch_msssim.MS_SSIM for the actual computation.
    Input range: [0, 1] (raw MRI intensity space).
    """

    def __init__(self, kernel_size=11, sigma=1.5, scales=5):
        super().__init__()
        self.ms_ssim = MS_SSIM(
            win_size=kernel_size,
            win_sigma=sigma,
            data_range=1.0,
            size_average=True,
            channel=1,
            spatial_dims=2,
        )

    def forward(self, x, y):
        return 1.0 - self.ms_ssim(x, y)


__all__ = [
    "MSSSIMLoss",
]
