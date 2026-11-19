from .src_coattention import CrossDirection, SpatialRelationCoAttention, SRCDetectionModel, SRCTrainer
from .neck import NeckSpatialRelationCoAttention


def register_modules() -> None:
    """Make the neck module available to the Ultralytics YAML parser."""
    import ultralytics.nn.tasks as tasks

    tasks.NeckSpatialRelationCoAttention = NeckSpatialRelationCoAttention


register_modules()

__all__ = [
    "CrossDirection",
    "SpatialRelationCoAttention",
    "SRCDetectionModel",
    "SRCTrainer",
    "NeckSpatialRelationCoAttention",
    "register_modules",
]
