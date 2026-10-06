"""Lightweight domain-conditioned residual adapter for RF RF endpoints."""

from __future__ import annotations

import torch
from torch import nn


class EndpointAligner(nn.Module):
    """Map real inversion endpoints toward a reference-domain distribution.

    The final convolution is zero initialized, so the initial mapping is
    exactly identity.  Domain labels condition the correction rather than
    redefining the endpoint itself.
    """

    def __init__(
        self,
        endpoint_channels=4,
        hidden_channels=24,
        domain_embed_dim=8,
        num_domains=15,
        num_blocks=2,
    ):
        super().__init__()
        self.endpoint_channels = int(endpoint_channels)
        hidden_channels = int(hidden_channels)
        norm_groups = min(8, hidden_channels)
        while hidden_channels % norm_groups:
            norm_groups -= 1
        self.domain_embedding = nn.Embedding(
            int(num_domains), int(domain_embed_dim)
        )
        input_channels = self.endpoint_channels + int(domain_embed_dim)
        self.input = nn.Conv3d(input_channels, hidden_channels, 3, padding=1)
        blocks = []
        for _ in range(int(num_blocks)):
            blocks.extend(
                [
                    nn.GroupNorm(norm_groups, hidden_channels),
                    nn.SiLU(),
                    nn.Conv3d(
                        hidden_channels,
                        hidden_channels,
                        3,
                        padding=1,
                    ),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv3d(
            hidden_channels, self.endpoint_channels, 3, padding=1
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, endpoint, domain_label):
        if endpoint.ndim != 5:
            raise ValueError(
                "endpoint must have shape [B,C,H,W,D], got "
                f"{tuple(endpoint.shape)}"
            )
        label = domain_label.to(
            device=endpoint.device, dtype=torch.long
        )
        embedding = self.domain_embedding(label)
        embedding = embedding.view(
            endpoint.shape[0], embedding.shape[1], 1, 1, 1
        ).expand(-1, -1, *endpoint.shape[2:])
        hidden = self.input(torch.cat([endpoint, embedding], dim=1))
        delta = self.output(self.blocks(hidden))
        return endpoint + delta, delta
