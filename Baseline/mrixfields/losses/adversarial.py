"""Adversarial losses for GAN training."""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """GAN loss supporting vanilla, least-squares, and hinge variants.

    Args:
        mode: Loss type - 'vanilla' (BCE), 'lsgan' (MSE), or 'hinge'.
        real_label: Target value for real samples (ignored for hinge).
        fake_label: Target value for fake samples (ignored for hinge).
    """

    def __init__(
        self,
        mode: str = "lsgan",
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))

        if mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode == "hinge":
            pass
        else:
            raise ValueError(f"Unknown GAN loss mode: {mode}")

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        if self.mode == "hinge":
            if is_real:
                return torch.relu(1.0 - prediction).mean()
            else:
                return torch.relu(1.0 + prediction).mean()
        target = prediction.new_full(prediction.shape, self.real_label.item() if is_real else self.fake_label.item())
        return self.loss(prediction, target)

    def g_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """Generator adversarial loss.

        Hinge: -E[D(G(z))]. Vanilla/lsgan: E[(D(G(z)) - 1)^2].
        """
        if self.mode == "hinge":
            return -prediction.mean()
        return self(prediction, True)

__all__ = [
    "GANLoss",
]
