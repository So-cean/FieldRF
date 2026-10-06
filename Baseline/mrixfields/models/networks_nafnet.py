"""NAFNet-based generator for CUT contrastive learning.

Reference: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022.

Adapted for CUT: adds forward(x, layers, encode_only) interface for NCE loss.

Architecture (official NAFNet U-Net, bilinear up / stride-2 down):
  Input → Conv3×3 (head)
    Encoder L1 × e1 (ngf, H×W)                                    → skip1
      → Conv3×3 s2 (→2*ngf)
    Encoder L2 × e2 (2*ngf, H/2×W/2)                                → skip2
      → Conv3×3 s2 (→4*ngf)
    Encoder L3 × e3 (4*ngf, H/4×W/4)                                → skip3
      → Conv3×3 s2 (→8*ngf)
    Encoder L4 × e4 (8*ngf, H/8×W/8)                                → skip4
      → Conv3×3 s2 (→16*ngf)
    Middle × m (16*ngf, H/16×W/16)                                   ← encode_only cut-off
      → Upsample(bilinear) + Conv3×3 (→8*ngf) + skip4
    Decoder L4 × d4 (8*ngf, H/8×W/8)
      → Upsample(bilinear) + Conv3×3 (→4*ngf) + skip3
    Decoder L3 × d3 (4*ngf, H/4×W/4)
      → Upsample(bilinear) + Conv3×3 (→2*ngf) + skip2
    Decoder L2 × d2 (2*ngf, H/2×W/2)
      → Upsample(bilinear) + Conv3×3 (→ngf) + skip1
    Decoder L1 × d1 (ngf, H×W)
  → Conv3×3 (→output_nc) + Tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Official NAFBlock: dual-branch with SCA, beta/gamma scaling.

    Branch 1: LN → Conv1×1 → DWConv3×3 → SimpleGate → SCA → Conv1×1 → Dropout
    Branch 2: LN → Conv1×1 → SimpleGate → Conv1×1 → Dropout
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_ch = c * DW_Expand

        # Branch 1
        self.conv1 = nn.Conv2d(c, dw_ch, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1,
                               groups=dw_ch, bias=True)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, kernel_size=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, kernel_size=1, bias=True),
        )
        self.sg = SimpleGate()
        self.norm1 = nn.LayerNorm(c)

        # Branch 2
        ffn_ch = c * FFN_Expand
        self.conv4 = nn.Conv2d(c, ffn_ch, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, kernel_size=1, bias=True)
        self.norm2 = nn.LayerNorm(c)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x):
        inp = x

        # Branch 1
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.dropout(x) * self.beta

        # Branch 2
        x = self.norm2(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.dropout(x) * self.gamma


class NAFNetGenerator(nn.Module):
    """NAFNet encoder-decoder generator with CUT-compatible forward interface.

    Parameters follow the official NAFNet paper:
        input_nc (int): input channels
        output_nc (int): output channels
        ngf (int): base channel count (default 32)
        enc_blk_nums (list): encoder blocks per level [E1, E2, E3, E4]
        middle_blk_num (int): middle bottleneck blocks
        dec_blk_nums (list): decoder blocks per level [D1, D2, D3, D4]
        DW_Expand (int): expansion factor for DWConv branch
        FFN_Expand (int): expansion factor for FFN branch
        drop_out_rate (float): dropout rate (default 0)
        use_tanh (bool): apply Tanh at output
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=32,
                 enc_blk_nums=None, middle_blk_num=12, dec_blk_nums=None,
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., use_tanh=True):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]
        self.use_tanh = use_tanh

        kw = dict(DW_Expand=DW_Expand, FFN_Expand=FFN_Expand, drop_out_rate=drop_out_rate)

        # Head
        self.head = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=True)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = ngf
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, **kw) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=3, stride=2, padding=1, bias=True))
            chan = chan * 2

        # Middle
        self.middle = nn.Sequential(*[NAFBlock(chan, **kw) for _ in range(middle_blk_num)])

        # Decoder (reverse order)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chan, chan // 2, kernel_size=3, padding=1, bias=True),
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, **kw) for _ in range(num)]))

        # Tail
        self.tail = nn.Conv2d(chan, output_nc, kernel_size=3, padding=1, bias=True)

    def forward(self, x, layers=None, encode_only=False):
        """Forward with optional intermediate feature extraction for CUT NCE loss.

        Args:
            x: Input tensor (B, C, H, W).
            layers: List of layer indices to extract features from.
                    [1, 2, 3, 4] mapped to: encoder L1, L2, L3, L4.
                    Empty or None → full forward.
            encode_only: If True, stop after middle and return features.

        Returns:
            If layers is empty/none: output tensor.
            If encode_only: list of intermediate feature tensors.
        """
        _, _, H, W = x.shape
        ph = (16 - H % 16) % 16
        pw = (16 - W % 16) % 16
        if ph > 0 or pw > 0:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode='reflect')

        if not layers:
            layers = []
        feats = []

        x = self.head(x)

        # Encoder (no skip storage)
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x = encoder(x)
            idx = i + 1
            if idx in layers:
                feats.append(x)
            x = down(x)

        # Middle
        x = self.middle(x)
        if 5 in layers:
            feats.append(x)

        if encode_only:
            return feats

        # Decoder (no skip connections)
        for up, decoder in zip(self.ups, self.decoders):
            x = up(x)
            x = decoder(x)

        x = self.tail(x)
        if self.use_tanh:
            x = torch.tanh(x)
        if ph > 0 or pw > 0:
            x = x[:, :, ph // 2:ph // 2 + H, pw // 2:pw // 2 + W]
        return x


__all__ = [
    "NAFBlock",
    "NAFNetGenerator",
]
