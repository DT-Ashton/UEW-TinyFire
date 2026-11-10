"""Lightweight spatial and relational co-attention for YOLO feature maps."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialRelationalCoAttentionLite(nn.Module):
    """Refine one fused feature map with spatial and pooled relational cues.

    Spatial evidence from the original feature map gates the pooled query, key,
    and value tokens. Relational context controls spatial and channel gates;
    their signed modulation can enhance relevant responses and suppress noise.
    This is attention within the fused map, not explicit pairwise attention
    across its source maps. Zero residual gain starts as exact identity. The
    attention matrix has at most ``pool_size ** 4`` elements per image.
    """

    def __init__(self, channels: int, reduction: int = 8, pool_size: int = 8):
        super().__init__()
        for name, value in (("channels", channels), ("reduction", reduction), ("pool_size", pool_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        self.channels = channels
        self.pool_size = pool_size
        hidden = max(8, channels // reduction)

        # The two descriptors share one inexpensive full-resolution spatial gate.
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.qkv = nn.Conv2d(channels, 3 * hidden, kernel_size=1, bias=False)
        self.relation_spatial = nn.Conv2d(hidden, 1, kernel_size=1, bias=False)
        self.relation_channel = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"SpatialRelationalCoAttentionLite expects [B, {self.channels}, H, W], "
                f"got {tuple(x.shape)}"
            )

        batch, _, height, width = x.shape
        pooled_size = (min(height, self.pool_size), min(width, self.pool_size))

        spatial_input = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        spatial_logits = self.spatial(spatial_input)
        spatial_tokens = F.adaptive_avg_pool2d(spatial_logits.sigmoid(), pooled_size)
        pooled = F.adaptive_avg_pool2d(x, pooled_size)

        # A single projection produces all Q/K/V tensors from spatially gated
        # tokens. Pooling Q as well as K/V bounds the quadratic attention cost.
        q, k, v = self.qkv(pooled * (1 + spatial_tokens)).flatten(2).chunk(3, dim=1)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        relation = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        relation = relation.transpose(1, 2).reshape(batch, -1, *pooled_size)
        relation_spatial = self.relation_spatial(relation)
        if pooled_size != (height, width):
            relation_spatial = F.interpolate(
                relation_spatial, size=(height, width), mode="bilinear", align_corners=False
            )

        spatial_gate = (spatial_logits + relation_spatial).sigmoid()
        channel_gate = self.relation_channel(F.adaptive_avg_pool2d(relation, 1)).sigmoid()
        modulation = spatial_gate + channel_gate - 1.0
        return x + self.gain.tanh() * (x * modulation)


__all__ = ["SpatialRelationalCoAttentionLite"]
