"""Parser-compatible weighted BiFPN fusion and cheap refinement blocks."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class WeightedBiFPNFusion(nn.Module):
    """Fuse equally sized features with normalized, learned positive weights.

    The preceding YAML node must concatenate ``inputs`` tensors along their
    channel axis. This module returns a one-item tuple for Ultralytics' built-in
    ``Index`` node to extract. ``Index`` also sets the parser's output channel
    count to ``channels`` without patching the installed parser.
    """

    def __init__(self, channels: int, inputs: int, epsilon: float = 1e-4):
        super().__init__()
        for name, value in (("channels", channels), ("inputs", inputs)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if inputs < 2:
            raise ValueError(f"inputs must be at least 2, got {inputs}")
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"epsilon must be a positive finite number, got {epsilon!r}")

        self.channels = channels
        self.inputs = inputs
        self.epsilon = float(epsilon)
        self.weights = nn.Parameter(torch.ones(inputs))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.inputs * self.channels:
            raise ValueError(
                f"WeightedBiFPNFusion expects [B, {self.inputs * self.channels}, H, W], "
                f"got {tuple(x.shape)}"
            )

        positive = F.softplus(self.weights)
        weights = positive / (positive.sum() + self.epsilon)
        features = x.split(self.channels, dim=1)
        fused = features[0] * weights[0]
        for feature, weight in zip(features[1:], weights[1:]):
            fused = fused + feature * weight
        return (fused,)


class BiFPNRefine(nn.Module):
    """Depthwise separable spatial refinement after each fusion node."""

    def __init__(self, channels: int):
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError(f"channels must be a positive integer, got {channels!r}")
        self.channels = channels
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"BiFPNRefine expects [B, {self.channels}, H, W], got {tuple(x.shape)}")
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


__all__ = ["WeightedBiFPNFusion", "BiFPNRefine"]
