"""Trainable MAISI decoder path, independent of the frozen encoding path."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class VAEDecoder(nn.Module):
    """Exactly the MAISI decode forward, including its latent projection.

    These are references to the original VAE's submodules. Keeping this
    callable outside the VAE preserves all autoencoder checkpoint keys while
    allowing DDP to synchronize every parameter used by decoder adaptation.
    """

    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder
        self.use_checkpoint = vae.use_checkpoint

    def forward(self, latent):
        latent = self.post_quant_conv(latent)
        if self.use_checkpoint:
            return checkpoint(self.decoder, latent, use_reentrant=False)
        return self.decoder(latent)


def prepare_decoder(vae):
    """Freeze encoder and encoding projections, then enable the decode path."""
    vae.requires_grad_(False)
    decoder = VAEDecoder(vae)
    decoder.requires_grad_(True)
    return decoder
