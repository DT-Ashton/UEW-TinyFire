import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics.nn.tasks as tasks

class SpatialRelationalCoAttention(nn.Module):
    """
    Spatial-Relational Co-Attention (SRC) Module
    Spatial-Relational Co-Attention nhẹ cho feature map P2.
    Query giữ độ phân giải P2; Key/Value được giảm không gian để tiết kiệm VRAM.
    """
    def __init__(self, c1=32, reduction=8, sr_ratio=4):
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"c1 must be positive, got {c1}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        if sr_ratio <= 0:
            raise ValueError(f"sr_ratio must be positive, got {sr_ratio}")

        self.c1 = c1
        self.inter_c = max(16, c1 // reduction)
        self.sr_ratio = sr_ratio

        # Query: giữ chi tiết không gian ở độ phân giải P2
        self.query_proj = nn.Sequential(
            nn.Conv2d(c1, self.inter_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_c),
            nn.SiLU()
        )

        # Key: mã hóa texture/ngữ cảnh cục bộ bằng tích chập 3x3
        self.key_proj = nn.Sequential(
            nn.Conv2d(c1, self.inter_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_c),
            nn.SiLU()
        )

        # Value được chiếu về cùng số chiều với Query/Key
        self.value_proj = nn.Conv2d(c1, self.inter_c, kernel_size=1, bias=False)

        # Spatial reduction cho Key/Value. sr_ratio=8 an toàn hơn cho P2 160x160 trên T4.
        if sr_ratio > 1:
            self.key_reduce = nn.Sequential(
                nn.Conv2d(self.inter_c, self.inter_c, kernel_size=sr_ratio, stride=sr_ratio, bias=False),
                # GroupNorm also works when P2 reduces to a 1x1 map and batch=1.
                nn.GroupNorm(1, self.inter_c)
            )
            self.value_reduce = nn.Sequential(
                nn.Conv2d(self.inter_c, self.inter_c, kernel_size=sr_ratio, stride=sr_ratio, bias=False),
                nn.GroupNorm(1, self.inter_c)
            )
        else:
            self.key_reduce = nn.Identity()
            self.value_reduce = nn.Identity()

        # Đưa attention output từ inter_c trở lại c1 channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.inter_c, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.c1:
            raise RuntimeError(f'SRC expected {self.c1} channels, but received {C}')

        # scaled_dot_product_attention dùng dạng [B, heads, tokens, dim]
        q = self.query_proj(x).flatten(2).transpose(1, 2).unsqueeze(1)

        k_map = self.key_reduce(self.key_proj(x))
        v_map = self.value_reduce(self.value_proj(x))
        k = k_map.flatten(2).transpose(1, 2).unsqueeze(1)
        v = v_map.flatten(2).transpose(1, 2).unsqueeze(1)

        # PyTorch tự chọn attention kernel tiết kiệm bộ nhớ khi GPU hỗ trợ.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.squeeze(1).transpose(1, 2).reshape(B, self.inter_c, H, W)

        return x + self.gamma * self.out_conv(out)


# Ultralytics resolves YAML module names from ``ultralytics.nn.tasks``.
# Registering at import time keeps the custom layer available to every model
# construction path, including the trainer's fresh model instance.
tasks.SpatialRelationalCoAttention = SpatialRelationalCoAttention

__all__ = ["SpatialRelationalCoAttention"]
