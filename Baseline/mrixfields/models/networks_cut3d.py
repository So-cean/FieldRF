"""3D network architectures for CUT and CycleGAN baselines.

Architecture mirrors networks.py but uses 3D convolutions for volumetric data.
"""

import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .net_init import init_net, init_weights


class Identity(nn.Module):
    """Identity pass-through used as a no-op normalization layer."""

    def forward(self, x):
        return x


def get_norm_layer_3d(norm_type="instance"):
    """Return a 3D normalization layer."""
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class Normalize(nn.Module):
    """L2 normalization along a given dimension."""

    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        return F.normalize(x, p=self.power, dim=1, eps=1e-7)


# --------------------------------------------------------------------------- #
#  3D Generator
# --------------------------------------------------------------------------- #

class ResnetGenerator3D(nn.Module):
    """Resnet-based 3D generator with optional encode_only mode for CUT NCE loss.

    Architecture: c7s1-ngf, d-2ngf, d-4ngf, R×n_blocks, u-2ngf, u-ngf, c7s1-output, Tanh
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=64,
                 norm_layer=nn.InstanceNorm3d, use_dropout=False,
                 n_blocks=9, padding_type="reflect", use_tanh=True,
                 skip_dropout: float = 0.0,
                 use_checkpoint: bool = False):
        assert n_blocks >= 0
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.use_checkpoint = use_checkpoint
        self.skip_drop = skip_dropout

        model = [
            nn.ReflectionPad3d(3),
            nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock3D(ngf * mult, padding_type=padding_type,
                                    norm_layer=norm_layer, use_dropout=use_dropout,
                                    use_bias=use_bias, skip_drop=self.skip_drop)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                          kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0, bias=True)]
        if use_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        """Forward pass with optional intermediate feature extraction for CUT.

        Args:
            input: 5D tensor (B, C, D, H, W).
            layers: List of layer indices to extract features from.
            encode_only: If True, return only intermediate features.

        Returns:
            output tensor or tuple (output, features).
        """
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                if self.use_checkpoint and isinstance(layer, nn.Conv3d):
                    feat = checkpoint(layer, feat, use_reentrant=False)
                else:
                    feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            if self.use_checkpoint:
                feat = input
                for layer in self.model:
                    if isinstance(layer, nn.Conv3d):
                        feat = checkpoint(layer, feat, use_reentrant=False)
                    else:
                        feat = layer(feat)
                return feat
            return self.model(input)


class ResnetBlock3D(nn.Module):
    """Define a 3D Resnet block with skip connections."""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, skip_drop=0.0):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.skip_drop = skip_drop

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        if self.training and self.skip_drop > 0 and torch.rand(1).item() < self.skip_drop:
            return self.conv_block(x)
        return x + self.conv_block(x)


# --------------------------------------------------------------------------- #
#  3D Discriminator
# --------------------------------------------------------------------------- #

class NLayerDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator.

    Parameters:
        input_nc (int) -- the number of channels in input images
        ndf (int)      -- the number of filters in the first conv layer
        n_layers (int) -- the number of conv layers in the discriminator
        norm_layer     -- normalization layer
    """

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d,
                 use_spectral_norm=False):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        sn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        kw = 4
        padw = 1
        sequence = [sn(nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                sn(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                             kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            sn(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [sn(nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        with torch.autocast(device_type=input.device.type, enabled=False):
            x = input.float()
            feats = []
            for layer in self.model:
                x = layer(x)
                feats.append(x)
            return feats  # list[Tensor], feats[-1] is prediction logit


# --------------------------------------------------------------------------- #
#  PatchSampleF3D — MLP projection head for CUT NCE loss
# --------------------------------------------------------------------------- #

class PatchSampleF3D(nn.Module):
    """Feature sampling + MLP projection for PatchNCE loss (3D volumes).

    Samples spatial patches from 3D feature maps and projects them through
    per-layer MLPs for contrastive learning.
    """

    def __init__(self, use_mlp=True, nc=256, init_type="normal", init_gain=0.02):
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, feats):
        """Lazily create MLP heads based on feature dimensions."""
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc),
            )
            mlp.to(feat.device)
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init_weights(self, self.init_type, self.init_gain)
        self.mlp_init = True

    def forward(self, feats, num_patches=256, patch_ids=None, mask=None):
        """Sample patches from 3D features and project through MLPs.

        Args:
            feats: List of 5D feature maps (B, C, D, H, W) from encoder layers.
            num_patches: Number of spatial patches to sample per layer.
            patch_ids: Pre-computed patch indices (for consistency between src/tgt).
            mask: Optional (B, 1, D, H, W) foreground mask for weighted sampling.

        Returns:
            return_feats: List of projected feature vectors.
            return_ids: List of sampled patch indices.
        """
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, C, D, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)  # B, D*H*W, C
            if num_patches > 0 and num_patches < feat_reshape.shape[1]:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    if mask is not None:
                        # Foreground-weighted sampling
                        m = F.interpolate(mask, size=(D, H, W), mode='nearest')
                        m = m.flatten(1) + 1e-8
                        weights = m[0] / m[0].sum()
                        patch_id = torch.multinomial(weights, num_patches, replacement=False)
                    else:
                        patch_id = np.random.permutation(feat_reshape.shape[1])
                        patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                    patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], D, H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids

__all__ = [
    "ResnetGenerator3D",
    "NLayerDiscriminator3D",
    "PatchSampleF3D",
]
