"""Shared discriminators: NLayerDiscriminator, MultiscaleDiscriminator."""

import torch
import torch.nn as nn

from .networks_cut3d import NLayerDiscriminator3D


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field with default n_layers=3).

    Parameters:
        input_nc (int) -- the number of channels in input images
        ndf (int)      -- the number of filters in the first conv layer
        n_layers (int) -- the number of conv layers in the discriminator
        norm_layer     -- normalization layer
        use_spectral_norm (bool) -- if True, apply spectral norm to all conv layers
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_spectral_norm=False):
        super().__init__()
        if type(norm_layer).__name__ == 'partial':
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1

        sn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        sequence = [
            sn(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                sn(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            sn(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [sn(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        with torch.autocast(device_type=input.device.type, enabled=False):
            x = input.float()
            feats = []
            for layer in self.model:
                x = layer(x)
                feats.append(x)
            return feats  # list[Tensor], feats[-1] is prediction logit


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator (pix2pixHD style).

    Stacks num_D identical NLayerDiscriminators at different resolutions.
    Each sub-discriminator sees the input at 1/2^i resolution via avg pooling.

    Parameters:
        input_nc (int) -- the number of channels in input images
        ndf (int)      -- the number of filters in the first conv layer
        n_layers (int) -- layers per sub-discriminator
        num_D (int)    -- number of sub-discriminators (scales)
        norm_layer     -- normalization layer
        use_spectral_norm (bool) -- if True, apply spectral norm to all conv layers
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3, num_D=2,
                 norm_layer=nn.InstanceNorm2d, use_spectral_norm=False):
        super().__init__()
        self.num_D = num_D
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer,
                                       use_spectral_norm=use_spectral_norm)
            setattr(self, f"layer_{i}", netD)
        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False
        )

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            netD = getattr(self, f"layer_{i}")
            result.append(netD(input))
            if i != self.num_D - 1:
                input = self.downsample(input)
        return result


class MultiscaleDiscriminator3D(nn.Module):
    """Multi-scale 3D PatchGAN discriminator.

    Stacks num_D identical NLayerDiscriminator3D at different resolutions.
    Each sub-discriminator sees the input at 1/2^i resolution via avg pooling.

    Parameters:
        input_nc (int) -- the number of channels in input images
        ndf (int)      -- the number of filters in the first conv layer
        n_layers (int) -- layers per sub-discriminator
        num_D (int)    -- number of sub-discriminators (scales)
        norm_layer     -- normalization layer
        use_spectral_norm (bool) -- if True, apply spectral norm
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3, num_D=2,
                 norm_layer=nn.InstanceNorm3d, use_spectral_norm=False):
        super().__init__()
        self.num_D = num_D
        for i in range(num_D):
            netD = NLayerDiscriminator3D(input_nc, ndf, n_layers, norm_layer,
                                         use_spectral_norm=use_spectral_norm)
            setattr(self, f"layer_{i}", netD)
        self.downsample = nn.AvgPool3d(
            3, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            netD = getattr(self, f"layer_{i}")
            result.append(netD(input))
            if i != self.num_D - 1:
                input = self.downsample(input)
        return result


__all__ = [
    "NLayerDiscriminator",
    "MultiscaleDiscriminator",
    "MultiscaleDiscriminator3D",
]