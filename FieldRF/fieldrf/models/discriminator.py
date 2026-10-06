"""Optional source-conditioned latent discriminator and GAN objectives."""

import torch
from torch import nn
import torch.nn.functional as F
from fieldrf.utils.domains import NUM_CLASSES


class LatentProjectionDiscriminator(nn.Module):
    """3D PatchGAN with projection conditioning."""

    def __init__(self, in_channels=4, num_classes=NUM_CLASSES, base_ch=64,
                 use_spectral_norm=False):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, base_ch * 4)

        def _sn(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        self.conv1 = nn.Sequential(
            _sn(nn.Conv3d(in_channels, base_ch, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            _sn(nn.Conv3d(base_ch, base_ch * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm3d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            _sn(nn.Conv3d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm3d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = _sn(nn.Conv3d(base_ch * 4, 1, 4, padding=1))

    def forward(self, z, class_label):
        f1 = self.conv1(z)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        patch_logits = self.head(f3)
        feat_pooled = f3.mean(dim=[2, 3, 4])
        class_vec = self.class_emb(class_label)
        proj_logits = (feat_pooled * class_vec).sum(dim=1, keepdim=True)
        return patch_logits, proj_logits, [f1, f2, f3]

    def full_logits(self, z, class_label):
        patch, proj, _ = self.forward(z, class_label)
        return patch + proj.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def lsgan_d_loss(real_logits, fake_logits):
    real_target = torch.ones_like(real_logits)
    fake_target = torch.zeros_like(fake_logits)
    return 0.5 * (F.mse_loss(real_logits, real_target) +
                  F.mse_loss(fake_logits, fake_target))


def lsgan_g_loss(fake_logits):
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))

