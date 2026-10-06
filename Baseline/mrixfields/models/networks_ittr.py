"""ITTR (Image-to-Image Translation with Transformers) networks.

Based on the official lucidrains implementation:
    https://github.com/lucidrains/ITTR-pytorch

Components:
    - ChanLayerNorm: channel-wise layer normalization
    - DPSA: Dual-Pruned Self-Attention
    - HPB: Hybrid Perception Block (DPSA + DWConv + FFN)
    - ITTRGenerator: encoder-bottleneck-decoder generator using HPB blocks

All ``einops`` operations have been replaced with native PyTorch to avoid
an extra dependency.
"""

import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_init import init_weights


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def l2norm(t):
    return F.normalize(t, dim=-1)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# --------------------------------------------------------------------------- #
#  DPSA — Dual-pruned Self-attention Block
# --------------------------------------------------------------------------- #

class DPSA(nn.Module):
    """Dual-pruned Self-attention Block.

    Uses top-k selection along height and width to prune the attention
    computation, making full self-attention feasible on high-res feature maps.
    """

    def __init__(
        self,
        dim,
        height_top_k=16,
        width_top_k=16,
        dim_head=32,
        heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = ChanLayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # fold out heads: (b, heads*dim_head, h, w) -> (b*heads, dim_head, h, w)
        q = q.view(b, self.heads, self.dim_head, h, w).reshape(b * self.heads, self.dim_head, h, w)
        k = k.view(b, self.heads, self.dim_head, h, w).reshape(b * self.heads, self.dim_head, h, w)
        v = v.view(b, self.heads, self.dim_head, h, w).reshape(b * self.heads, self.dim_head, h, w)

        # l2 normalized queries and keys (cosine sim attention)
        q, k = l2norm(q), l2norm(k)

        need_height_select = self.height_top_k < h
        need_width_select = self.width_top_k < w

        if need_height_select or need_width_select:
            # q_probe: sum over spatial dims -> (b*heads, dim_head)
            q_probe = q.sum(dim=(2, 3))

        # gather along height
        if need_height_select:
            # k_height: (b*heads, dim_head, h)
            k_height = k.sum(dim=3)
            # scores: (b*heads, h)
            scores_h = torch.bmm(q_probe.unsqueeze(1), k_height).squeeze(1)
            top_h_indices = scores_h.topk(k=self.height_top_k, dim=-1).indices  # (b*heads, top_k)
            # expand indices for gather along height (dim=2)
            top_h_indices = top_h_indices.unsqueeze(1).unsqueeze(-1)  # (b*heads, 1, top_k, 1)
            top_h_indices = top_h_indices.expand(-1, self.dim_head, -1, w)  # (b*heads, dim_head, top_k, w)
            k = k.gather(2, top_h_indices)
            v = v.gather(2, top_h_indices)

        # gather along width
        if need_width_select:
            # k_width: (b*heads, dim_head, w)
            k_width = k.sum(dim=2)
            scores_w = torch.bmm(q_probe.unsqueeze(1), k_width).squeeze(1)  # (b*heads, w)
            top_w_indices = scores_w.topk(k=self.width_top_k, dim=-1).indices  # (b*heads, top_k)
            # expand indices for gather along width (dim=3)
            top_w_indices = top_w_indices.unsqueeze(1).unsqueeze(2)  # (b*heads, 1, 1, top_k)
            top_w_indices = top_w_indices.expand(-1, self.dim_head, k.shape[2], -1)  # (b*heads, dim_head, h', top_k)
            k = k.gather(3, top_w_indices)
            v = v.gather(3, top_w_indices)

        # flatten spatial: (b*heads, h*w, dim_head)
        q = q.reshape(b * self.heads, -1, self.dim_head)
        k = k.reshape(b * self.heads, -1, self.dim_head)
        v = v.reshape(b * self.heads, -1, self.dim_head)

        # cosine similarities
        sim = torch.bmm(q, k.transpose(1, 2))

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)

        # merge heads and reshape back: (b*heads, h*w, dim_head) -> (b, heads*dim_head, h, w)
        out = out.view(b, self.heads, h, w, self.dim_head)
        out = out.permute(0, 1, 4, 2, 3).contiguous()
        out = out.view(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)


# --------------------------------------------------------------------------- #
#  HPB — Hybrid Perception Block
# --------------------------------------------------------------------------- #

class HPB(nn.Module):
    """Hybrid Perception Block.

    Combines a DPSA branch with a depth-wise conv branch in parallel,
    followed by a feed-forward network (FFN).
    """

    def __init__(
        self,
        dim,
        dim_head=32,
        heads=8,
        ff_mult=4,
        attn_height_top_k=16,
        attn_width_top_k=16,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()

        self.attn = DPSA(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            height_top_k=attn_height_top_k,
            width_top_k=attn_width_top_k,
            dropout=attn_dropout,
        )

        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.attn_parallel_combine_out = nn.Conv2d(dim * 2, dim, 1)

        ff_inner_dim = int(dim * ff_mult)

        self.ff = nn.Sequential(
            nn.Conv2d(dim, ff_inner_dim, 1),
            nn.InstanceNorm2d(ff_inner_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            Residual(nn.Sequential(
                nn.Conv2d(ff_inner_dim, ff_inner_dim, 3, padding=1, groups=ff_inner_dim),
                nn.InstanceNorm2d(ff_inner_dim),
                nn.GELU(),
                nn.Dropout(ff_dropout),
            )),
            nn.Conv2d(ff_inner_dim, dim, 1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        attn_branch = self.attn(x)
        conv_branch = self.dwconv(x)

        concatted = torch.cat((attn_branch, conv_branch), dim=1)
        attn_out = self.attn_parallel_combine_out(concatted) + x

        return self.ff(attn_out)


# --------------------------------------------------------------------------- #
#  ITTR Generator
# --------------------------------------------------------------------------- #

class ITTRGenerator(nn.Module):
    """ITTR Generator: CNN encoder → HPB bottleneck → CNN decoder.

    Compatible with CUT NCE loss via ``encode_only`` and ``layers``.
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        n_blocks=9,
        dim_head=32,
        heads=8,
        attn_height_top_k=16,
        attn_width_top_k=16,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        use_tanh=True,
        use_conv_transpose=False,
        norm_layer=nn.InstanceNorm2d,
    ):
        super().__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # ---- Encoder --------------------------------------------------------
        encoder_layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder_layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                          stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        self.encoder = nn.Sequential(*encoder_layers)

        # ---- Bottleneck: HPB blocks -----------------------------------------
        bottleneck_dim = ngf * (2 ** n_downsampling)
        self.hpb_blocks = nn.ModuleList([
            HPB(
                dim=bottleneck_dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                attn_height_top_k=attn_height_top_k,
                attn_width_top_k=attn_width_top_k,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
            for _ in range(n_blocks)
        ])

        # ---- Decoder --------------------------------------------------------
        decoder_layers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            in_ch = ngf * mult
            out_ch = int(ngf * mult / 2)
            if use_conv_transpose:
                decoder_layers += [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                       padding=1, output_padding=1, bias=use_bias),
                    norm_layer(out_ch),
                    nn.ReLU(True),
                ]
            else:
                decoder_layers += [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                              bias=use_bias),
                    norm_layer(out_ch),
                    nn.ReLU(True),
                ]

        decoder_layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        ]
        if use_tanh:
            decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, input, layers=[], encode_only=False):
        """Forward with optional intermediate feature extraction for CUT.

        Layer index mapping (for default n_blocks=9):
            0: after initial conv      (ngf)
            1: after first downsample  (ngf*2)
            2: after second downsample (ngf*4)
            3..(3+n_blocks-1): after each HPB block (ngf*4)
            (3+n_blocks): after first upsample  (ngf*2)
            ... etc.
        """
        if -1 in layers:
            layers.append(len(self.encoder) + len(self.hpb_blocks) + len(self.decoder))

        if len(layers) > 0:
            feat = input
            feats = []

            # Encoder
            for layer_id, layer in enumerate(self.encoder):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats

            # HPB blocks
            base_idx = len(self.encoder)
            for i, hpb in enumerate(self.hpb_blocks):
                feat = hpb(feat)
                layer_id = base_idx + i
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats

            # Decoder
            base_idx = len(self.encoder) + len(self.hpb_blocks)
            for layer_id, layer in enumerate(self.decoder):
                feat = layer(feat)
                global_id = base_idx + layer_id
                if global_id in layers:
                    feats.append(feat)
                if global_id == layers[-1] and encode_only:
                    return feats

            return feat, feats
        else:
            x = self.encoder(input)
            for hpb in self.hpb_blocks:
                x = hpb(x)
            x = self.decoder(x)
            return x


# --------------------------------------------------------------------------- #
#  Shared components
# --------------------------------------------------------------------------- #

class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        return F.normalize(x, p=self.power, dim=1, eps=1e-7)


class PatchSampleF(nn.Module):
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
    "ITTRGenerator",
]
