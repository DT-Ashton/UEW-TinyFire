"""
Central configuration, dataclasses, and constants for fire_audit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import yaml

# Class definitions
CLASS_MAP: Dict[int, str] = {0: "fire", 1: "smoke"}
NAME_TO_CLASS_ID: Dict[str, int] = {"fire": 0, "smoke": 1}
CLASS_NAMES: List[str] = ["fire", "smoke"]
CLASS_IDS: Set[int] = {0, 1}

# Numerical tolerance for normalized coordinates
DEFAULT_EPSILON: float = 1e-4

# Valid file extensions
VALID_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_LABEL_EXTENSIONS: Set[str] = {".txt"}

# Artifact directories are kept inside each dataset root so outputs from
# different datasets cannot overwrite one another.
MANIFEST_DIR_NAME: str = "manifest"
SPLITS_DIR_NAME: str = "splits"

# The editable runtime configuration is kept at the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_USER_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


def get_user_config_path(config_path: Optional[Path] = None) -> Path:
    """Return the project config path, honoring an environment override."""
    if config_path is not None:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_USER_CONFIG_PATH


def load_user_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the shared external YAML config, or return an empty config."""
    path = get_user_config_path(config_path)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"User config must contain a YAML mapping: {path}")
    return data


def resolve_user_path(value: Optional[Any], config_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve a path value relative to the project root."""
    if value is None or str(value).strip() == "":
        return None
    raw = os.path.expandvars(str(value)).replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_configured_data_path(config_path: Optional[Path] = None) -> Optional[Path]:
    """Return the configured dataset root or YOLO data YAML path."""
    config = load_user_config(config_path)
    paths = config.get("paths", {})
    value = config.get("data_path")
    if value is None and isinstance(paths, dict):
        value = paths.get("data") or paths.get("data_path")
    return resolve_user_path(value, config_path=config_path)


def get_configured_data_yaml_path(config_path: Optional[Path] = None) -> Optional[Path]:
    """Return a usable YOLO data YAML path from ``data_path``."""
    data_path = get_configured_data_path(config_path)
    if data_path is None:
        return None
    if data_path.suffix.lower() in {".yaml", ".yml"}:
        return data_path
    for candidate in (data_path / MANIFEST_DIR_NAME / "data.yaml", data_path / "data.yaml"):
        if candidate.is_file():
            return candidate.resolve()
    return data_path / MANIFEST_DIR_NAME / "data.yaml"


def get_configured_data_root(config_path: Optional[Path] = None) -> Optional[Path]:
    """Return a dataset directory derived from the configured data path."""
    data_path = get_configured_data_path(config_path)
    if data_path is None:
        return None
    if data_path.is_file() or data_path.suffix.lower() in {".yaml", ".yml"}:
        return data_path.parent.parent if data_path.parent.name == MANIFEST_DIR_NAME else data_path.parent
    return data_path


def get_training_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Return training settings with configured paths resolved."""
    config = load_user_config(config_path)
    paths = config.get("paths", {})
    training = config.get("training", {})
    settings = dict(training) if isinstance(training, dict) else {}
    weights = settings.get("weights") or settings.get("weights_path")
    if weights is None and isinstance(paths, dict):
        weights = paths.get("weights") or paths.get("weights_path")
    resolved_weights = resolve_user_path(weights, config_path=config_path)
    if resolved_weights is not None:
        settings["weights"] = resolved_weights
    model_yaml = settings.get("model_yaml") or settings.get("model")
    resolved_model_yaml = resolve_user_path(model_yaml, config_path=config_path)
    if resolved_model_yaml is not None:
        settings["model_yaml"] = resolved_model_yaml
    data_path = get_configured_data_yaml_path(config_path)
    if data_path is not None:
        settings.setdefault("data", data_path)
    return settings


def get_configured_weights_path(config_path: Optional[Path] = None) -> Optional[Path]:
    """Return the checkpoint path configured for training/evaluation/inference."""
    settings = get_training_config(config_path)
    return resolve_user_path(settings.get("weights"), config_path=config_path)


def get_configured_inference_source(config_path: Optional[Path] = None) -> Optional[Path]:
    """Return the optional inference source configured by the user."""
    config = load_user_config(config_path)
    paths = config.get("paths", {})
    value = config.get("inference_source")
    if value is None and isinstance(paths, dict):
        value = paths.get("inference_source") or paths.get("source")
    return resolve_user_path(value, config_path=config_path)


@dataclass
class BoundingBox:
    """Represents a normalized YOLO bounding box."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_tuple(self) -> Tuple[int, float, float, float, float]:
        """Return box coordinates as a standard 5-tuple."""
        return (self.class_id, self.x_center, self.y_center, self.width, self.height)

    def to_yolo_str(self) -> str:
        """Format box as a standard YOLO annotation line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def pixel_area(self, img_width: int, img_height: int) -> float:
        """Calculate pixel area of bounding box given image dimensions."""
        return (self.width * img_width) * (self.height * img_height)

    def scale_category(self, img_width: int, img_height: int) -> str:
        """
        Categorize box scale according to standard COCO metrics:
        - Small: area < 32^2 (1024 px^2)
        - Medium: 32^2 <= area <= 96^2 (1024 - 9216 px^2)
        - Large: area > 96^2 (9216 px^2)
        """
        if img_width <= 0 or img_height <= 0:
            return "unknown"
        area = self.pixel_area(img_width, img_height)
        if area < 32 ** 2:
            return "small (<32^2)"
        elif area <= 96 ** 2:
            return "medium (32^2-96^2)"
        else:
            return "large (>96^2)"


@dataclass
class ImageRecord:
    """Represents a single image and its corresponding annotation metadata."""
    image_path: Path
    label_path: Optional[Path] = None
    dataset: str = ""
    split_source: str = ""
    sequence_id: str = ""
    boxes: List[BoundingBox] = field(default_factory=list)
    width: int = 0
    height: int = 0
    is_negative: bool = False
    is_corrupt: bool = False
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating a YOLO label file or annotation line."""
    valid: bool
    boxes: List[BoundingBox] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_negative: bool = False


@dataclass
class DatasetScanResult:
    """Result of scanning a dataset directory."""
    dataset_name: str
    root_path: Path
    records: List[ImageRecord] = field(default_factory=list)
    missing_labels: List[Path] = field(default_factory=list)
    corrupt_images: List[Path] = field(default_factory=list)
    corrupt_labels: List[Path] = field(default_factory=list)


@dataclass
class DatasetStats:
    """Aggregated statistical metrics for a dataset."""
    dataset_name: str
    total_images: int = 0
    valid_images: int = 0
    annotated_images: int = 0
    empty_frames: int = 0
    corrupt_files: int = 0
    total_boxes: int = 0
    class_distribution: Dict[str, int] = field(default_factory=lambda: {"fire": 0, "smoke": 0})
    co_occurrence: Dict[str, int] = field(default_factory=lambda: {
        "fire_only": 0,
        "smoke_only": 0,
        "both_fire_and_smoke": 0,
        "neither_background": 0,
    })
    resolutions: Dict[str, int] = field(default_factory=dict)
    aspect_ratios: Dict[str, int] = field(default_factory=dict)
    box_scale_distribution: Dict[str, int] = field(default_factory=lambda: {
        "small (<32^2)": 0,
        "medium (32^2-96^2)": 0,
        "large (>96^2)": 0,
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to serializable dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_images": self.total_images,
            "valid_images": self.valid_images,
            "annotated_images": self.annotated_images,
            "empty_frames": self.empty_frames,
            "corrupt_files": self.corrupt_files,
            "total_boxes": self.total_boxes,
            "class_distribution": self.class_distribution,
            "co_occurrence": self.co_occurrence,
            "resolutions": self.resolutions,
            "aspect_ratios": self.aspect_ratios,
            "box_scale_distribution": self.box_scale_distribution,
        }


@dataclass
class SegmentedFrame:
    """Represents a single frame categorized as video frame or static image."""
    filename: str
    path: Path
    category: Literal["video_frame", "static_image"]
    video_id: Optional[str] = None          # e.g., "bothFireAndSmoke_video_0001" or None
    frame_index: Optional[int] = None       # 0-indexed within video sequence
    sequence_length: Optional[int] = None   # Total length of containing video sequence
    width: int = 0
    height: int = 0
    aspect_ratio: str = ""
    fire_boxes: int = 0
    smoke_boxes: int = 0
    is_negative: bool = False
    is_corrupt: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return dict representation suitable for JSON manifest serialization."""
        return {
            "filename": self.filename,
            "path": str(self.path).replace("\\", "/"),
            "category": self.category,
            "video_id": self.video_id,
            "frame_index": self.frame_index,
            "sequence_length": self.sequence_length,
            "resolution": {"width": self.width, "height": self.height},
            "boxes": {"fire": self.fire_boxes, "smoke": self.smoke_boxes},
            "is_negative": self.is_negative,
        }

    def to_csv_row(self) -> Dict[str, Any]:
        """Return flat dict suitable for CSV manifest writing."""
        return {
            "filename": self.filename,
            "path": str(self.path).replace("\\", "/"),
            "category": self.category,
            "video_id": self.video_id or "",
            "frame_index": self.frame_index if self.frame_index is not None else "",
            "sequence_length": self.sequence_length if self.sequence_length is not None else "",
            "width": self.width,
            "height": self.height,
            "fire_boxes": self.fire_boxes,
            "smoke_boxes": self.smoke_boxes,
            "is_negative": self.is_negative,
        }


@dataclass
class VideoSequence:
    """Represents a detected contiguous video sequence."""
    video_id: str
    category_prefix: str
    start_index: int
    end_index: int
    total_frames: int
    resolution: Tuple[int, int]
    frames: List[SegmentedFrame] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable summary of video sequence."""
        return {
            "video_id": self.video_id,
            "category_prefix": self.category_prefix,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "total_frames": self.total_frames,
            "resolution": {"width": self.resolution[0], "height": self.resolution[1]},
            "frame_filenames": [f.filename for f in self.frames],
        }


@dataclass
class SegmentationResult:
    """Outcome of dataset video sequence segmentation."""
    total_images: int = 0
    video_frames_count: int = 0
    static_images_count: int = 0
    total_video_sequences: int = 0
    sequences: Dict[str, VideoSequence] = field(default_factory=dict)
    records: List[SegmentedFrame] = field(default_factory=list)
    manifest_json_path: Optional[Path] = None
    manifest_csv_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert segmentation result summary to dictionary."""
        return {
            "total_images": self.total_images,
            "video_frames_count": self.video_frames_count,
            "static_images_count": self.static_images_count,
            "total_video_sequences": self.total_video_sequences,
            "manifest_json_path": str(self.manifest_json_path).replace("\\", "/") if self.manifest_json_path else None,
            "manifest_csv_path": str(self.manifest_csv_path).replace("\\", "/") if self.manifest_csv_path else None,
        }


@dataclass
class SegmentationConfig:
    """Configuration options for video sequence segmentation engine."""
    min_sequence_length: int = 5
    dhash_threshold: int = 18
    lookahead_k: int = 2
    smooth_ratio_threshold: float = 0.60
    max_workers: int = 16
    dataset_path: Optional[Path] = None
    output_dir: Optional[Path] = None

