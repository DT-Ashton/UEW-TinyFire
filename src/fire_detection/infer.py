"""
Inference and Visual Demonstration Pipeline for Fire and Smoke Detection.
Runs object detection on images/directories and renders bounding boxes with labels and confidences.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.modules.bifpn_src import register_modules as register_bifpn_src_modules

register_bifpn_src_modules()

try:
    from src.fire_audit.config import get_configured_inference_source, get_configured_weights_path
except ImportError:
    from fire_audit.config import get_configured_inference_source, get_configured_weights_path

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Visual styling colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 69, 255),    # Fire: Bright Red-Orange
    1: (235, 206, 135), # Smoke: Sky Blue / Cyan
    "fire": (0, 69, 255),
    "smoke": (235, 206, 135),
    "default": (0, 255, 0),
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, defaulting on None, NaN, Inf, or type conversion errors."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_coord(val: Any, default: int = 0) -> int:
    """Safely convert coordinate to integer, handling NaN, Inf, and type errors."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return int(round(f))
    except (ValueError, TypeError):
        return default


def read_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Robustly read image across all platforms including Windows unicode paths."""
    p = Path(path).resolve()
    if not p.is_file():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"Error reading image {p}: {e}")
        return None


def write_image(path: Union[str, Path], img: np.ndarray) -> bool:
    """Robustly write image across all platforms including Windows unicode paths."""
    p = Path(path).resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix if p.suffix else ".jpg"
        success, buffer = cv2.imencode(ext, img)
        if success:
            with open(p, "wb") as f:
                f.write(buffer)
            return True
        return False
    except Exception as e:
        logger.warning(f"Error writing image {p}: {e}")
        return False


def draw_bounding_box(
    img: np.ndarray,
    box: Tuple[float, float, float, float],
    label: str,
    score: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a styled bounding box with label tag on the image."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return img

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if w <= 0 or h <= 0:
        return img

    # Parse and order coordinates safely
    coords = [_safe_coord(v, default=0) for v in box]
    x1, y1, x2, y2 = coords
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Clamp coordinates to image boundaries
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Safe score formatting
    score_val = _safe_float(score, default=0.0)
    text = f"{label} {score_val:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    tag_h = text_h + baseline + 4
    tag_w = text_w + 6

    # Adjust horizontal position if close to right edge
    tag_x1 = max(0, min(x1, w - tag_w - 1)) if w > tag_w else 0
    tag_x2 = min(w - 1, tag_x1 + tag_w)

    # Adjust vertical position: if box near top, draw inside box; else draw above box
    if y1 - tag_h >= 0:
        tag_y1 = y1 - tag_h
        tag_y2 = y1
        text_y = y1 - baseline - 2
    else:
        tag_y1 = y1
        tag_y2 = min(h - 1, y1 + tag_h)
        text_y = min(h - 1, y1 + text_h + 2)

    # Filled background tag for legibility
    cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    # Text in dark or white depending on contrast
    text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 150 else (255, 255, 255)
    cv2.putText(
        img,
        text,
        (tag_x1 + 3, max(0, text_y)),
        font,
        font_scale,
        text_color,
        text_thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def infer_yolo(
    weights: Optional[Union[str, Path]] = None,
    source: Optional[Union[str, Path]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: Optional[Union[str, int]] = None,
    output_dir: Union[str, Path] = "runs/infer",
    save_txt: bool = False,
    json_out: Optional[Union[str, Path]] = None,
    max_images: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run inference on images, save visualizations, and return detection summary.

    Returns:
        Dict containing summary of processed images, detections per image, and detection counts.
    """
    weights = weights if weights is not None else get_configured_weights_path()
    if weights is None:
        raise ValueError("Inference weights are not configured; set training.weights in the config.yaml")
    weights_path = Path(weights).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    source = source if source is not None else get_configured_inference_source()
    if source is None:
        raise ValueError("Inference source is not configured; set paths.inference_source in the config.yaml")
    source_path = Path(source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Inference source not found: {source_path}")

    # Validate parameters
    if conf < 0.0 or conf > 1.0:
        raise ValueError(f"Confidence threshold conf must be between 0.0 and 1.0, got {conf}")
    if iou < 0.0 or iou > 1.0:
        raise ValueError(f"IoU threshold iou must be between 0.0 and 1.0, got {iou}")
    if imgsz < 32:
        raise ValueError(f"imgsz must be at least 32, got {imgsz}")
    if max_images is not None and max_images < 1:
        raise ValueError(f"max_images must be a positive integer if specified, got {max_images}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    device_str = str(device)

    # Resolve list of input images before loading model
    image_files: List[Path] = []
    if source_path.is_file():
        if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            raise ValueError(
                f"Unsupported image file format '{source_path.suffix}' for source {source_path}. "
                f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTS)}"
            )
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = sorted([p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS])

    if len(image_files) == 0:
        raise ValueError(f"No supported images found in source: {source_path}")

    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]

    logger.info(f"Loading YOLO model from: {weights_path}")
    model = YOLO(str(weights_path))

    logger.info(f"Processing {len(image_files)} image(s) on device {device_str}...")

    total_detections = 0
    class_detection_counts: Dict[str, int] = {"fire": 0, "smoke": 0}
    detections_by_image: List[Dict[str, Any]] = []

    names_map = getattr(model, "names", {0: "fire", 1: "smoke"})
    if not isinstance(names_map, dict):
        names_map = {i: str(n) for i, n in enumerate(names_map)}

    for idx, img_p in enumerate(image_files, start=1):
        orig_img = read_image(img_p)
        if orig_img is None:
            logger.warning(f"Could not load or decode image: {img_p}")
            detections_by_image.append({
                "image_name": img_p.name,
                "image_path": str(img_p),
                "output_path": None,
                "num_detections": 0,
                "detections": [],
                "error": "Failed to decode image file",
            })
            continue

        # Run inference
        results = model.predict(
            source=str(img_p),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device_str,
            verbose=verbose,
        )

        res = results[0]
        img_detections: List[Dict[str, Any]] = []
        txt_lines: List[str] = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes_data = res.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            for box_row in boxes_data:
                x1, y1, x2, y2, score, cls_id_float = box_row[:6]
                cls_id = int(cls_id_float)
                cls_name = names_map.get(cls_id, f"class_{cls_id}")
                color = CLASS_COLORS.get(cls_id, CLASS_COLORS.get(cls_name, CLASS_COLORS["default"]))

                # Draw bounding box
                draw_bounding_box(orig_img, (x1, y1, x2, y2), cls_name, score, color)

                img_detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(_safe_float(score), 4),
                    "box_xyxy": [round(_safe_float(v), 2) for v in (x1, y1, x2, y2)],
                })

                class_detection_counts[cls_name] = class_detection_counts.get(cls_name, 0) + 1
                total_detections += 1

                # YOLO normalized txt format if requested
                if save_txt:
                    h, w = orig_img.shape[:2]
                    if w > 0 and h > 0:
                        xc = ((x1 + x2) / 2.0) / w
                        yc = ((y1 + y2) / 2.0) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        safe_score = _safe_float(score)
                        txt_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {safe_score:.4f}")

        # Save annotated image
        out_img_path = out_dir / img_p.name
        write_image(out_img_path, orig_img)

        # Save label txt if requested
        if save_txt:
            txt_dir = out_dir / "labels"
            txt_dir.mkdir(parents=True, exist_ok=True)
            (txt_dir / f"{img_p.stem}.txt").write_text("\n".join(txt_lines), encoding="utf-8")

        detections_by_image.append({
            "image_name": img_p.name,
            "image_path": str(img_p),
            "output_path": str(out_img_path),
            "num_detections": len(img_detections),
            "detections": img_detections,
        })

    summary: Dict[str, Any] = {
        "status": "SUCCESS",
        "model": str(weights_path),
        "source": str(source_path),
        "output_directory": str(out_dir),
        "total_images_processed": len(image_files),
        "total_detections": total_detections,
        "class_detection_counts": class_detection_counts,
        "images": detections_by_image,
    }

    if json_out is not None:
        json_p = Path(json_out).resolve()
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved inference results JSON to: {json_p}")

    logger.info(
        f"Inference complete: {len(image_files)} images processed, "
        f"{total_detections} detections saved to {out_dir}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for inference."""
    parser = argparse.ArgumentParser(
        prog="infer",
        description="Run inference with YOLOv8 fire/smoke model and save visual detections",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to fine-tuned model checkpoint (default: training.weights from config.yaml)")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source image, directory, or video (default: paths.inference_source from config.yaml)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", "--img-size", dest="imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (default: auto)")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", type=str, default="runs/infer", help="Output directory for saved visualizations (default: runs/infer)")
    parser.add_argument("--save-txt", action="store_true", help="Save detection bounding boxes to txt files")
    parser.add_argument("--json-out", type=str, default=None, help="Path to save output JSON predictions report")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process (optional)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for inference."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        infer_yolo(
            weights=args.weights,
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=args.output_dir,
            save_txt=args.save_txt,
            json_out=args.json_out,
            max_images=args.max_images,
        )
        return 0
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
