"""Neck-only spatial and bidirectional relational attention for YOLO features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _grid_xy(height: int, width: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height,
        (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width,
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


class _CrossDirection(nn.Module):
    """Read pooled source tokens with full-resolution target queries."""

    def __init__(self, dim: int, heads: int, grid: int, chunk: int):
        super().__init__()
        self.heads = heads
        self.grid = grid
        self.chunk = chunk
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, bias=False)
        self.distance_scale = nn.Parameter(torch.zeros(heads))

    def forward(
        self, target: torch.Tensor, source: torch.Tensor, source_gate: torch.Tensor
    ) -> torch.Tensor:
        batch, dim, height, width = target.shape
        grid_size = (min(self.grid, height), min(self.grid, width))
        source_small = F.adaptive_avg_pool2d(source, grid_size)
        gate_small = F.adaptive_avg_pool2d(source_gate, grid_size)

        def tokens(feature: torch.Tensor) -> torch.Tensor:
            return (
                feature.flatten(2)
                .transpose(1, 2)
                .reshape(batch, -1, self.heads, dim // self.heads)
                .transpose(1, 2)
            )

        query = tokens(self.q(target))
        key = tokens(self.k(source_small))
        value = tokens(self.v(source_small))
        query_xy = _grid_xy(height, width, target.device)
        key_xy = _grid_xy(*grid_size, target.device)
        log_gate = gate_small.float().flatten(2).clamp_min(1e-4).log().unsqueeze(1)
        distance_scale = F.softplus(self.distance_scale.float())[None, :, None, None]

        chunks = []
        for start in range(0, height * width, self.chunk):
            distance = (query_xy[start : start + self.chunk, None] - key_xy[None]).square().sum(-1)
            bias = (-distance_scale * distance[None, None] + log_gate).to(query.dtype)
            chunks.append(
                F.scaled_dot_product_attention(
                    query[:, :, start : start + self.chunk], key, value, attn_mask=bias, dropout_p=0.0
                )
            )

        context = (
            torch.cat(chunks, dim=2)
            .transpose(1, 2)
            .reshape(batch, height * width, dim)
            .transpose(1, 2)
            .reshape(batch, dim, height, width)
        )
        presence = source_gate.amax(dim=(2, 3), keepdim=True)
        return self.out(context) * presence


class NeckSpatialRelationCoAttention(nn.Module):
    """Exchange information between latent fire and smoke branches in the neck.

    This follows the two-direction attention core in ``src_coattention.py``.
    The gates are learned from detection loss alone; they are not supervised
    fire/smoke occupancy maps without the sample's auxiliary trainer.
    """

    def __init__(self, channels: int, dim: int = 32, heads: int = 4, grid: int = 8, chunk: int = 1024):
        super().__init__()
        for name, value in (("channels", channels), ("dim", dim), ("heads", heads), ("grid", grid), ("chunk", chunk)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if dim % heads or dim % 4:
            raise ValueError("dim must be divisible by heads and by 4 for GroupNorm")

        self.channels = channels
        self.fire = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.GroupNorm(4, dim), nn.SiLU())
        self.smoke = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.GroupNorm(4, dim), nn.SiLU())
        self.fire_gate = nn.Conv2d(dim, 1, 1)
        self.smoke_gate = nn.Conv2d(dim, 1, 1)
        self.fire_from_smoke = _CrossDirection(dim, heads, grid, chunk)
        self.smoke_from_fire = _CrossDirection(dim, heads, grid, chunk)
        self.fuse = nn.Conv2d(2 * dim, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"NeckSpatialRelationCoAttention expects [B, {self.channels}, H, W], got {tuple(x.shape)}"
            )

        fire = self.fire(x)
        smoke = self.smoke(x)
        fire_gate = self.fire_gate(fire).sigmoid()
        smoke_gate = self.smoke_gate(smoke).sigmoid()
        fire = fire * fire_gate
        smoke = smoke * smoke_gate
        fire_context = self.fire_from_smoke(fire, smoke, smoke_gate)
        smoke_context = self.smoke_from_fire(smoke, fire, fire_gate)
        context = torch.cat((fire_context * fire_gate, smoke_context * smoke_gate), dim=1)
        return x + self.gamma * self.fuse(context)


__all__ = ["NeckSpatialRelationCoAttention"]
