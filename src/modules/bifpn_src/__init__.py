"""Reusable BiFPN and spatial-relational layers for the custom YOLO26 model."""

from .attention import SpatialRelationalCoAttentionLite
from .fusion import BiFPNRefine, WeightedBiFPNFusion


def register_modules() -> None:
    """Expose custom layer names to the Ultralytics YAML parser."""
    import ultralytics.nn.tasks as tasks

    tasks.SpatialRelationalCoAttentionLite = SpatialRelationalCoAttentionLite
    tasks.WeightedBiFPNFusion = WeightedBiFPNFusion
    tasks.BiFPNRefine = BiFPNRefine


register_modules()

__all__ = [
    "SpatialRelationalCoAttentionLite",
    "WeightedBiFPNFusion",
    "BiFPNRefine",
    "register_modules",
]
