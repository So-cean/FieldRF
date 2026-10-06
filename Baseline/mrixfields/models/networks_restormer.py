"""Restormer-based generator for CUT contrastive learning.

Reference: Zamir et al., "Restormer: Efficient Transformer for
High-Resolution Image Restoration", CVPR 2022.

Adapted for CUT: adds forward(x, layers, encode_only) interface for NCE loss.

Architecture (4-level encoder-decoder + bottleneck, bilinear up / stride-2 down):
  Input → Conv3×3 (head)
    Encoder L1 × b1 (ngf, H×W)                              → skip1
      → Conv3×3 s2 (→2*ngf)
    Encoder L2 × b2 (2*ngf, H/2×W/2)                         → skip2
      → Conv3×3 s2 (→4*ngf)
    Encoder L3 × b3 (4*ngf, H/4×W/4)                         → skip3
      → Conv3×3 s2 (→8*ngf)
    Bottleneck × b4 (8*ngf, H/8×W/8)                         ← encode_only cut-off
      → Upsample(bilinear) + Conv3×3 (→4*ngf) + cat(skip3) + Conv1×1
    Decoder L3 × b3 (4*ngf, H/4×W/4)
      → Upsample(bilinear) + Conv3×3 (→2*ngf) + cat(skip2) + Conv1×1
    Decoder L2 × b2 (2*ngf, H/2×W/2)
      → Upsample(bilinear) + Conv3×3 (→ngf) + cat(skip1)
    Decoder L1 × b1 (2*ngf, H×W)
  → Conv3×3 (→output_nc) + Tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer)."""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.dw_qkv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1,
                                groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.dw_qkv(self.project_qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        heads = self.num_heads
        head_dim = C // heads
        q = q.reshape(B, heads, head_dim, H * W)
        k = k.reshape(B, heads, head_dim, H * W)
        v = v.reshape(B, heads, head_dim, H * W)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (Restormer)."""

    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                 groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dw_conv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class RestormerBlock(nn.Module):
    """Restormer block: LayerNorm → MDTA → +x → LayerNorm → GDFN → +x."""

    def __init__(self, channels, num_heads=8, expansion_factor=2.66):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.mdta = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.gdfn = GDFN(channels, expansion_factor)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.mdta(x) + shortcut

        shortcut = x
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.gdfn(x) + shortcut
        return x





class RestormerGenerator(nn.Module):
    """Restormer encoder-decoder generator with CUT-compatible forward interface.

    Layer indices for NCE:
        1=E1, 2=E2, 3=E3, 4=bottleneck (encoder)
        5=D3, 6=D2, 7=D1              (decoder)

    Parameters match the original Restormer paper:
        input_nc (int): input channels
        output_nc (int): output channels
        ngf (int): base channel count (default 48)
        encoder_blocks (list): blocks per stage [E1, E2, E3, bottleneck]
        use_tanh (bool): apply Tanh at output
        num_heads (list): attention heads per stage [E1, E2, E3, bottleneck]
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=48,
                 encoder_blocks=None, use_tanh=True,
                 num_heads=None):
        super().__init__()
        if encoder_blocks is None:
            encoder_blocks = [2, 2, 4, 6]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        b1, b2, b3, b4 = encoder_blocks
        h1, h2, h3, h4 = num_heads
        self.use_tanh = use_tanh

        # Head: 3×3 Conv
        self.head = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False)

        # Encoder
        self.encoder_1 = nn.Sequential(*[RestormerBlock(ngf, num_heads=h1) for _ in range(b1)])
        self.down_1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True)

        self.encoder_2 = nn.Sequential(*[RestormerBlock(ngf * 2, num_heads=h2) for _ in range(b2)])
        self.down_2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True)

        self.encoder_3 = nn.Sequential(*[RestormerBlock(ngf * 4, num_heads=h3) for _ in range(b3)])
        self.down_3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, bias=True)

        self.bottleneck = nn.Sequential(*[RestormerBlock(ngf * 8, num_heads=h4) for _ in range(b4)])

        # Decoder
        self.up_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, padding=1, bias=True),
        )
        self.reduce_3 = nn.Conv2d(ngf * 8, ngf * 4, kernel_size=1, bias=False)
        self.decoder_3 = nn.Sequential(*[RestormerBlock(ngf * 4, num_heads=h3) for _ in range(b3)])

        self.up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=1, bias=True),
        )
        self.reduce_2 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=1, bias=False)
        self.decoder_2 = nn.Sequential(*[RestormerBlock(ngf * 2, num_heads=h2) for _ in range(b2)])

        self.up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1, bias=True),
        )
        self.decoder_1 = nn.Sequential(*[RestormerBlock(ngf * 2, num_heads=h1) for _ in range(b1)])

        # Tail: 3×3 Conv (decoder_1 outputs 2*ngf)
        self.tail = nn.Conv2d(ngf * 2, output_nc, kernel_size=3, padding=1, bias=False)

    def forward(self, x, layers=None, encode_only=False):
        """Forward pass. Layers: 1=E1,2=E2,3=E3,4=BN,5=D3,6=D2,7=D1."""
        _, _, H, W = x.shape
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph > 0 or pw > 0:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode='reflect')

        if not layers:
            layers = []
        feats = []

        # Head
        x = self.head(x)

        # Encoder: loop over (layer_id, block, down)
        skips = []
        for lid, block, down in [
            (1, self.encoder_1, self.down_1),
            (2, self.encoder_2, self.down_2),
            (3, self.encoder_3, self.down_3),
        ]:
            x = block(x)
            if lid in layers:
                feats.append(x)
            skips.append(x)
            x = down(x)

        # Bottleneck (layer 4)
        x = self.bottleneck(x)
        if 4 in layers:
            feats.append(x)

        # Decoder — skip when encode_only and no decoder layers requested
        if not encode_only or any(l > 4 for l in layers):
            for lid, up, reduce, block in [
                (5, self.up_3, self.reduce_3, self.decoder_3),
                (6, self.up_2, self.reduce_2, self.decoder_2),
                (7, self.up_1, None,          self.decoder_1),
            ]:
                x = up(x)
                x = torch.cat([x, skips.pop()], dim=1)
                if reduce is not None:
                    x = reduce(x)
                x = block(x)
                if lid in layers:
                    feats.append(x)

        if encode_only:
            return feats

        x = self.tail(x)
        if self.use_tanh:
            x = torch.tanh(x)
        if ph > 0 or pw > 0:
            x = x[:, :, ph // 2:ph // 2 + H, pw // 2:pw // 2 + W]
        return x


class RestormerGeneratorNoSkip(nn.Module):
    """Restormer U-Net WITHOUT skip connections.

    Same encoder/decoder as RestormerGenerator, but decoder gets NO
    skip connections from the encoder.  Forces the decoder to rely
    entirely on the bottleneck features.

    Layer indices for NCE (same as RestormerGenerator):
        1=E1, 2=E2, 3=E3, 4=bottleneck (encoder)
        5=D3, 6=D2, 7=D1              (decoder)
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=48,
                 encoder_blocks=None, use_tanh=True,
                 num_heads=None):
        super().__init__()
        if encoder_blocks is None:
            encoder_blocks = [2, 2, 4, 6]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        b1, b2, b3, b4 = encoder_blocks
        h1, h2, h3, h4 = num_heads
        self.use_tanh = use_tanh

        # Head: 3×3 Conv
        self.head = nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=False)

        # Encoder
        self.encoder_1 = nn.Sequential(*[RestormerBlock(ngf, num_heads=h1) for _ in range(b1)])
        self.down_1 = nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True)

        self.encoder_2 = nn.Sequential(*[RestormerBlock(ngf * 2, num_heads=h2) for _ in range(b2)])
        self.down_2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True)

        self.encoder_3 = nn.Sequential(*[RestormerBlock(ngf * 4, num_heads=h3) for _ in range(b3)])
        self.down_3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, bias=True)

        self.bottleneck = nn.Sequential(*[RestormerBlock(ngf * 8, num_heads=h4) for _ in range(b4)])

        # Decoder (no skip connections, no reduce convs)
        self.up_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, padding=1, bias=True),
        )
        self.decoder_3 = nn.Sequential(*[RestormerBlock(ngf * 4, num_heads=h3) for _ in range(b3)])

        self.up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=1, bias=True),
        )
        self.decoder_2 = nn.Sequential(*[RestormerBlock(ngf * 2, num_heads=h2) for _ in range(b2)])

        self.up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=1, bias=True),
        )
        self.decoder_1 = nn.Sequential(*[RestormerBlock(ngf, num_heads=h1) for _ in range(b1)])

        # Tail: 3×3 Conv (decoder_1 outputs ngf)
        self.tail = nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1, bias=False)

    def forward(self, x, layers=None, encode_only=False):
        _, _, H, W = x.shape
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph > 0 or pw > 0:
            x = F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode='reflect')

        if not layers:
            layers = []
        feats = []

        # Head
        x = self.head(x)

        # Encoder (no skip storage)
        for lid, block, down in [
            (1, self.encoder_1, self.down_1),
            (2, self.encoder_2, self.down_2),
            (3, self.encoder_3, self.down_3),
        ]:
            x = block(x)
            if lid in layers:
                feats.append(x)
            x = down(x)

        # Bottleneck (layer 4)
        x = self.bottleneck(x)
        if 4 in layers:
            feats.append(x)

        # Decoder — no cat, no reduce
        if not encode_only or any(l > 4 for l in layers):
            for lid, up, block in [
                (5, self.up_3, self.decoder_3),
                (6, self.up_2, self.decoder_2),
                (7, self.up_1, self.decoder_1),
            ]:
                x = up(x)
                x = block(x)
                if lid in layers:
                    feats.append(x)

        if encode_only:
            return feats

        x = self.tail(x)
        if self.use_tanh:
            x = torch.tanh(x)
        if ph > 0 or pw > 0:
            x = x[:, :, ph // 2:ph // 2 + H, pw // 2:pw // 2 + W]
        return x


__all__ = [
    "RestormerBlock",
    "MDTA",
    "GDFN",
    "RestormerGenerator",
    "RestormerGeneratorNoSkip",
]
