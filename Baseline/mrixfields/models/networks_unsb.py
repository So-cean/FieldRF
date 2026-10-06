"""UNSB networks: conditional ResNet generator + conditional PatchGAN discriminator.

Adapted from:
    https://github.com/cyclomon/UNSB
    Kim et al., "Unpaired Image-to-Image Translation via Neural Schrödinger Bridge", ICLR 2024.

Key components:
    - ResnetGenerator_ncsn: time-conditional + latent-conditional ResNet generator
    - NLayerDiscriminator_ncsn: time-conditional PatchGAN discriminator
    - Energy network E: same arch as D, but takes concatenated [Xt, Xt_1] as input
"""

import functools
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_init import init_net, init_weights


###############################################################################
# Helpers
###############################################################################

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


class PixelNorm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1, eps=1e-5)


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
            act,
        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.main[0].in_features)
        return self.main(temb)


class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.style_net = nn.Linear(style_dim, in_channel * 2)
        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, x, style):
        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        return gamma * x + beta


###############################################################################
# Downsample / Upsample (antialiased)
###############################################################################

def _get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1.0])
    elif filt_size == 2:
        a = np.array([1.0, 1.0])
    elif filt_size == 3:
        a = np.array([1.0, 2.0, 1.0])
    elif filt_size == 4:
        a = np.array([1.0, 3.0, 3.0, 1.0])
    elif filt_size == 5:
        a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    else:
        raise ValueError(f"Unsupported filter size: {filt_size}")
    filt = torch.Tensor(a[:, None] * a[None, :])
    return filt / torch.sum(filt)


class Downsample(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [p + pad_off for p in self.pad_sizes]
        self.stride = stride
        self.channels = channels
        filt = _get_filter(filt_size)
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        self.pad = nn.ReflectionPad2d(self.pad_sizes) if pad_type == "reflect" else nn.ZeroPad2d(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            return inp[:, :, :: self.stride, :: self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=4, stride=2):
        super().__init__()
        self.filt_size = filt_size
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.channels = channels
        filt = _get_filter(filt_size) * (stride ** 2)
        self.register_buffer("filt", filt[None, None, :, :].repeat(channels, 1, 1, 1))
        # NPU bf16: ReplicationPad2d not supported for DT_BFLOAT16.
        # ReflectionPad2d is already used in Downsample and is bf16-safe.
        pads = [self.pad_size] * 4
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(pads)
        else:
            self.pad = nn.ZeroPad2d(pads)

    def forward(self, inp):
        ret = F.conv_transpose2d(
            self.pad(inp), self.filt, stride=self.stride,
            padding=1 + self.pad_size, groups=inp.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_size % 2 == 0:
            ret = ret[:, :, :-1, :-1]
        return ret


###############################################################################
# ResNet Block (conditional)
###############################################################################

class ResnetBlock_cond(nn.Module):
    """ResNet block conditioned on time embedding and latent z."""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim):
        super().__init__()
        self.conv_block = nn.ModuleList()
        p = 0
        if padding_type == "reflect":
            self.conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding {padding_type}")

        self.conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        self.conv_block.append(norm_layer(dim))
        self.adaptive = AdaptiveLayer(dim, z_dim)
        self.conv_fin = nn.ModuleList([nn.ReLU()])
        if use_dropout:
            self.conv_fin.append(nn.Dropout(0.5))

        if padding_type == "reflect":
            self.conv_fin.append(nn.ReflectionPad2d(1))
        elif padding_type == "zero":
            p = 1
        self.conv_fin.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        self.conv_fin.append(norm_layer(dim))

        self.dense_time = nn.Linear(temb_dim, dim)
        init.zeros_(self.dense_time.bias)

        self.style = nn.Linear(z_dim, dim * 2)
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

    def forward(self, x, time_cond, z):
        time_input = self.dense_time(time_cond)
        out = x
        for n, layer in enumerate(self.conv_block):
            out = layer(out)
            if n == 0:
                out = out + time_input[:, :, None, None]
        out = self.adaptive(out, z)
        for layer in self.conv_fin:
            out = layer(out)
        return x + out


###############################################################################
# Generator
###############################################################################

class ResnetGenerator_ncsn(nn.Module):
    """Conditional ResNet generator for UNSB.

    Inputs: (x, time_cond, z)
        x:         (B, C, H, W)
        time_cond: (B,)  integer timestep indices
        z:         (B, z_dim)  latent code
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
        n_blocks=9,
        n_mlp=3,
        z_dim=256,
        t_emb_dim=256,
    ):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ngf = ngf
        self.z_dim = z_dim

        # Encoder
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(),
                Downsample(ngf * mult * 2),
            ]
        self.model = nn.Sequential(*model)

        # Conditional ResNet blocks
        mult = 2 ** n_downsampling
        self.model_res = nn.ModuleList([
            ResnetBlock_cond(
                ngf * mult, padding_type="reflect", norm_layer=norm_layer,
                use_dropout=use_dropout, use_bias=use_bias,
                temb_dim=t_emb_dim, z_dim=z_dim,
            )
            for _ in range(n_blocks)
        ])

        # Decoder
        model_upsample = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_upsample += [
                Upsample(ngf * mult),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(),
            ]
        model_upsample += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model_upsample = nn.Sequential(*model_upsample)

        # Z mapping network (PixelNorm + MLP)
        mapping_layers = [PixelNorm(), nn.Linear(z_dim, z_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_mlp):
            mapping_layers.append(nn.Linear(z_dim, z_dim))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)

        # Time embedding MLP
        modules_emb = [
            nn.Linear(ngf, t_emb_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.LeakyReLU(0.2),
        ]
        init.zeros_(modules_emb[0].bias)
        init.zeros_(modules_emb[2].bias)
        self.time_embed = nn.Sequential(*modules_emb)

    def forward(self, x, time_cond, z, layers=None, encode_only=False):
        z_embed = self.z_transform(z)
        temb = get_timestep_embedding(time_cond, self.ngf)
        time_embed = self.time_embed(temb)

        layers = layers or []
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
            for layer_id, layer in enumerate(self.model_res):
                feat = layer(feat, time_embed, z_embed)
                global_id = len(self.model) + layer_id
                if global_id in layers:
                    feats.append(feat)
                if global_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            out = self.model(x)
            for layer in self.model_res:
                out = layer(out, time_embed, z_embed)
            out = self.model_upsample(out)
            return out


###############################################################################
# Discriminator (conditional)
###############################################################################

class ConvBlock_cond(nn.Module):
    def __init__(self, in_channel, out_channel, t_emb_dim, kernel_size=4, stride=1, padding=1,
                 norm_layer=None, downsample=True, use_bias=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=use_bias)
        self.use_norm = norm_layer is not None
        if self.use_norm:
            self.norm = norm_layer(out_channel)
        self.act = nn.LeakyReLU(0.2)
        self.down = Downsample(out_channel)
        self.dense = nn.Linear(t_emb_dim, out_channel)

    def forward(self, input, t_emb):
        out = self.conv1(input)
        out = out + self.dense(t_emb)[..., None, None]
        if self.use_norm:
            out = self.norm(out)
        out = self.act(out)
        if self.downsample:
            out = self.down(out)
        return out


class NLayerDiscriminator_ncsn(nn.Module):
    """Time-conditional PatchGAN discriminator for UNSB."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, t_emb_dim=256):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model_main = nn.ModuleList()
        self.model_main.append(
            ConvBlock_cond(input_nc, ndf, t_emb_dim, kernel_size=4, stride=1, padding=1, use_bias=use_bias)
        )

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.model_main.append(
                ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, t_emb_dim,
                               kernel_size=4, stride=1, padding=1, use_bias=use_bias, norm_layer=norm_layer)
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, t_emb_dim,
                           kernel_size=4, stride=1, padding=1, use_bias=use_bias,
                           norm_layer=norm_layer, downsample=False)
        )
        self.final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        self.t_embed = TimestepEmbedding(embedding_dim=t_emb_dim, hidden_dim=t_emb_dim,
                                         output_dim=t_emb_dim, act=nn.LeakyReLU(0.2))

    def forward(self, input, t_emb, input2=None):
        t_emb = self.t_embed(t_emb)
        if input2 is not None:
            out = torch.cat([input, input2], dim=1)
        else:
            out = input
        for layer in self.model_main:
            out = layer(out, t_emb)
        return self.final_conv(out)


###############################################################################
# PatchSampleF — MLP projection head for PatchNCE
###############################################################################

class Normalize(nn.Module):
    """L2 normalization along a given dimension."""

    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        return F.normalize(x, p=self.power, dim=1, eps=1e-7)


class PatchSampleF(nn.Module):
    """Feature sampling + MLP projection for PatchNCE loss.

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
                    weight = weight.view(B, -1).clamp(1e-4, 1.0) * 0.8 + 0.1
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
    "ResnetGenerator_ncsn",
    "NLayerDiscriminator_ncsn",
    "PatchSampleF",
]
