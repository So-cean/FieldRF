"""CUT networks: ResnetGenerator, ResnetBlock, PatchSampleF."""

import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_init import init_weights


class Identity(nn.Module):
    """Identity pass-through used as a no-op normalization layer."""

    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer.

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
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


class ResnetGenerator(nn.Module):
    """Resnet-based generator with optional encode_only mode for CUT NCE loss.

    Architecture: c7s1-ngf, d-2ngf, d-4ngf, R*n_blocks, u-2ngf, u-ngf, c7s1-output, Tanh

    Based on CycleGAN official with encode_only from CUT official.

    Parameters:
        input_nc (int)      -- the number of channels in input images (default 1 for MRI)
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers
        n_blocks (int)      -- the number of ResNet blocks
        padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 n_blocks=9, padding_type="reflect", use_tanh=True,
                 skip_dropout: float = 0.0):
        assert n_blocks >= 0
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Original 7×7 stem
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        self.skip_drop = skip_dropout

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, skip_drop=self.skip_drop)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                          kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=True)]
        if use_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        """Forward pass with optional intermediate feature extraction for CUT.

        Args:
            input: Input tensor.
            layers: List of layer indices to extract features from (for CUT NCE loss).
                   If empty, standard forward pass.
            encode_only: If True, return only intermediate features (stop after last requested layer).

        Returns:
            If layers is empty: output tensor.
            If layers is non-empty and encode_only: list of intermediate features.
            If layers is non-empty and not encode_only: (output, list of intermediate features).
        """
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block with skip connections."""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, skip_drop=0.0):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.skip_drop = skip_drop

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        if self.training and self.skip_drop > 0 and torch.rand(1).item() < self.skip_drop:
            return self.conv_block(x)
        return x + self.conv_block(x)


# --------------------------------------------------------------------------- #
#  PatchSampleF — MLP projection head for CUT NCE loss
# --------------------------------------------------------------------------- #

class PatchSampleF(nn.Module):
    """Feature sampling + MLP projection for PatchNCE loss.

    Based on CUT official. Samples spatial patches from feature maps and
    projects them through per-layer MLPs for contrastive learning.

    Parameters:
        use_mlp (bool) -- if True, use MLP projection heads
        nc (int)       -- output feature dimension (default 256)
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
        """Sample patches from features and project through MLPs.

        Args:
            feats: List of feature maps from encoder layers.
            num_patches: Number of spatial patches to sample per layer.
            patch_ids: Pre-computed patch indices (for consistency between src/tgt).
            mask: Foreground mask (B, 1, H_img, W_img), 1=foreground, 0=background.
                  When provided, sampling is weighted by foreground probability.

        Returns:
            return_feats: List of projected feature vectors.
            return_ids: List of sampled patch indices.
        """
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                    if isinstance(patch_id, torch.Tensor):
                        patch_id = patch_id.to(device=feat.device, dtype=torch.long)
                    else:
                        patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                    if patch_id.dim() == 2:
                        row_idx = torch.arange(B, device=feat.device).unsqueeze(1)
                        x_sample = feat_reshape[row_idx, patch_id]
                    else:
                        x_sample = feat_reshape[:, patch_id, :]
                    x_sample = x_sample.flatten(0, 1)
                elif mask is not None:
                    weight = F.interpolate(mask.float(), size=(H, W), mode="bilinear", align_corners=False)
                    weight = weight.view(B, -1).clamp(1e-4, 1.0) * 0.95 + 0.05
                    patch_id = torch.multinomial(weight, num_patches, replacement=False)
                    row_idx = torch.arange(B, device=feat.device).unsqueeze(1)
                    x_sample = feat_reshape[row_idx, patch_id]
                    x_sample = x_sample.flatten(0, 1)
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
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids

__all__ = [
    "ResnetGenerator",
    "PatchSampleF",
]
