import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics.nn.tasks as tasks


class SpatialRelationalCoAttention(nn.Module):
    """
    SRC with learnable 2D coordinate positional encoding.

    Query keeps the original spatial resolution.
    Key/Value are spatially reduced for lower attention cost.
    """

    def __init__(
        self,
        c1=32,
        reduction=8,
        sr_ratio=8,
        use_pos_encoding=True,
    ):
        super().__init__()

        if c1 <= 0:
            raise ValueError(f"c1 must be positive, got {c1}")
        if reduction <= 0:
            raise ValueError(
                f"reduction must be positive, got {reduction}"
            )
        if sr_ratio <= 0:
            raise ValueError(
                f"sr_ratio must be positive, got {sr_ratio}"
            )

        self.c1 = c1
        self.inter_c = max(16, c1 // reduction)
        self.sr_ratio = sr_ratio
        self.use_pos_encoding = use_pos_encoding

        self.query_proj = nn.Sequential(
            nn.Conv2d(c1, self.inter_c, 1, bias=False),
            nn.BatchNorm2d(self.inter_c),
            nn.SiLU(),
        )

        self.key_proj = nn.Sequential(
            nn.Conv2d(
                c1, self.inter_c,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.inter_c),
            nn.SiLU(),
        )

        self.value_proj = nn.Conv2d(
            c1, self.inter_c, 1, bias=False
        )

        if sr_ratio > 1:
            self.key_reduce = nn.Sequential(
                nn.Conv2d(
                    self.inter_c,
                    self.inter_c,
                    kernel_size=sr_ratio,
                    stride=sr_ratio,
                    bias=False,
                ),
                nn.GroupNorm(1, self.inter_c),
            )
            self.value_reduce = nn.Sequential(
                nn.Conv2d(
                    self.inter_c,
                    self.inter_c,
                    kernel_size=sr_ratio,
                    stride=sr_ratio,
                    bias=False,
                ),
                nn.GroupNorm(1, self.inter_c),
            )
        else:
            self.key_reduce = nn.Identity()
            self.value_reduce = nn.Identity()

        # Shared projection for normalized (x, y) coordinates.
        if use_pos_encoding:
            self.pos_proj = nn.Conv2d(
                2, self.inter_c, kernel_size=1, bias=False
            )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                self.inter_c, c1,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        # Identity initialization for stable fine-tuning.
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _coordinate_grid(H, W, device, dtype):
        """Return normalized 2D coordinates with shape [1, 2, H, W]."""
        y = torch.linspace(
            -1.0, 1.0, H, device=device, dtype=dtype
        )
        x = torch.linspace(
            -1.0, 1.0, W, device=device, dtype=dtype
        )

        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape

        if C != self.c1:
            raise RuntimeError(
                f"SRC expected {self.c1} channels, got {C}"
            )

        # Query: full-resolution P2.
        q_map = self.query_proj(x)

        # Key/Value: reduced spatial resolution.
        k_map = self.key_reduce(self.key_proj(x))
        v_map = self.value_reduce(self.value_proj(x))

        if self.use_pos_encoding:
            Hk, Wk = k_map.shape[-2:]

            q_coords = self._coordinate_grid(
                H, W, x.device, q_map.dtype
            )
            k_coords = self._coordinate_grid(
                Hk, Wk, x.device, k_map.dtype
            )

            # Add coordinate embeddings to Q and K only.
            q_map = q_map + self.pos_proj(q_coords)
            k_map = k_map + self.pos_proj(k_coords)

        # [B, 1, N, D]
        q = q_map.flatten(2).transpose(1, 2).unsqueeze(1)
        k = k_map.flatten(2).transpose(1, 2).unsqueeze(1)
        v = v_map.flatten(2).transpose(1, 2).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0
        )

        out = (
            out.squeeze(1)
            .transpose(1, 2)
            .reshape(B, self.inter_c, H, W)
        )

        return x + self.gamma * self.out_conv(out)


# Register the custom layer for Ultralytics YAML parsing.
tasks.SpatialRelationalCoAttention = SpatialRelationalCoAttention

__all__ = ["SpatialRelationalCoAttention"]
