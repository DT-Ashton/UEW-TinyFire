"""
Evaluation and Metrics Reporting Pipeline for Fire and Smoke Detection.
Computes standard detection metrics (Precision, Recall, mAP50, mAP50-95) and size-filtered AP,
and exports quantitative results to structured JSON and CSV formats.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import ap_per_class, box_iou

try:
    from src.fire_audit.config import get_configured_data_yaml_path, get_configured_weights_path
except ImportError:
    from fire_audit.config import get_configured_data_yaml_path, get_configured_weights_path

logger = logging.getLogger(__name__)

SIZE_THRESHOLDS = {"small": 32, "tiny": 16}


def _summarize_size_ap(records: Dict[str, List[np.ndarray]], names: Dict[int, str], niou: int) -> Dict[str, Any]:
    """Run Ultralytics AP once per IoU threshold so ignored detections are excluded."""
    arrays = {key: np.concatenate(values) for key, values in records.items()}
    counts = np.bincount(arrays["target_cls"].astype(int), minlength=len(names))
    ap = np.zeros((len(names), niou))
    if counts.sum():
        for i in range(niou):
            keep = ~arrays["ignored"][:, i]
            result = ap_per_class(
                arrays["tp"][keep, i : i + 1],
                arrays["conf"][keep],
                arrays["pred_cls"][keep],
                arrays["target_cls"],
            )
            ap[result[6], i] = result[5][:, 0]

    active = counts > 0
    per_class = {
        cname: {
            "map50": round(_safe_float(ap[int(cid), 0]), 5),
            "map50_95": round(_safe_float(ap[int(cid)].mean()), 5),
            "num_targets": int(counts[int(cid)]),
        }
        for cid, cname in names.items()
    }
    return {
        "overall": {
            "map50": round(_safe_float(ap[active, 0].mean()) if active.any() else 0.0, 5),
            "map50_95": round(_safe_float(ap[active].mean()) if active.any() else 0.0, 5),
            "num_targets": int(counts.sum()),
        },
        "per_class": per_class,
    }


class SizeAwareDetectionValidator(DetectionValidator):
    """Collect size-filtered AP alongside the unchanged YOLO detection metrics."""

    def init_metrics(self, model: Any) -> None:
        super().init_metrics(model)
        self.size_stats = {
            name: {key: [] for key in ("tp", "ignored", "conf", "pred_cls", "target_cls")}
            for name in SIZE_THRESHOLDS
        }

    def _size_matches(
        self, pred: Dict[str, torch.Tensor], batch: Dict[str, Any], min_side: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match eligible targets first, then mark one-to-one matches to ignored targets."""
        boxes = ops.scale_boxes(
            batch["imgsz"], batch["bboxes"].clone(), batch["ori_shape"], batch["ratio_pad"]
        )
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        eligible = area >= min_side ** 2
        selected = {**batch, "cls": batch["cls"][eligible], "bboxes": batch["bboxes"][eligible]}
        tp = self._process_batch(pred, selected)["tp"]
        ignored = np.zeros_like(tp)

        outside = ~eligible
        if outside.any() and pred["cls"].numel():
            ignored_iou = box_iou(batch["bboxes"][outside], pred["bboxes"])
            same_class = batch["cls"][outside][:, None] == pred["cls"]
            ignored_iou = (ignored_iou * same_class).cpu().numpy()
            for i, threshold in enumerate(self.iouv.cpu().tolist()):
                matches = np.argwhere((ignored_iou >= threshold) & ~tp[:, i][None, :])
                if len(matches):
                    # Mirror YOLO's IoU-first, one-to-one assignment for non-crowd labels.
                    matches = matches[ignored_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    ignored[matches[:, 1], i] = True

        return tp, ignored, selected["cls"].cpu().numpy()

    def update_metrics(self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            prepared = self._prepare_batch(si, batch)
            pred = self._prepare_pred(pred)
            conf = pred["conf"].cpu().numpy()
            pred_cls = pred["cls"].cpu().numpy()
            for name, min_side in SIZE_THRESHOLDS.items():
                tp, ignored, target_cls = self._size_matches(pred, prepared, min_side)
                records = self.size_stats[name]
                records["tp"].append(tp)
                records["ignored"].append(ignored)
                records["conf"].append(conf)
                records["pred_cls"].append(pred_cls)
                records["target_cls"].append(target_cls)

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        self.metrics.size_ap = {
            name: {
                "min_sqrt_area_px": min_side,
                **_summarize_size_ap(self.size_stats[name], self.names, self.niou),
            }
            for name, min_side in SIZE_THRESHOLDS.items()
        }
        return stats


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


def extract_metrics_dict(val_results: Any, split_name: str) -> Dict[str, Any]:
    """Extract standard detection metrics from Ultralytics validation results."""
    box = getattr(val_results, "box", None)
    speed = getattr(val_results, "speed", None) or {}

    names_dict = getattr(val_results, "names", {0: "fire", 1: "smoke"})
    if not isinstance(names_dict, dict):
        names_dict = {i: str(n) for i, n in enumerate(names_dict)}

    mp = _safe_float(getattr(box, "mp", 0.0)) if box is not None else 0.0
    mr = _safe_float(getattr(box, "mr", 0.0)) if box is not None else 0.0
    map50 = _safe_float(getattr(box, "map50", 0.0)) if box is not None else 0.0
    map50_95 = _safe_float(getattr(box, "map", 0.0)) if box is not None else 0.0
    fitness = _safe_float(getattr(val_results, "fitness", 0.0))

    # Per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    if box is not None and hasattr(box, "class_result") and hasattr(box, "ap_class_index"):
        for i, cls_idx in enumerate(box.ap_class_index):
            c_name = names_dict.get(int(cls_idx), names_dict.get(str(cls_idx), f"class_{cls_idx}"))
            try:
                res = box.class_result(i)
                # res is (p, r, ap50, ap)
                p_val, r_val, ap50_val, ap_val = res
                per_class[c_name] = {
                    "precision": round(_safe_float(p_val), 5),
                    "recall": round(_safe_float(r_val), 5),
                    "map50": round(_safe_float(ap50_val), 5),
                    "map50_95": round(_safe_float(ap_val), 5),
                }
            except Exception as e:
                logger.warning(f"Could not compute class result for {c_name}: {e}")

    # Ensure all target classes exist in dict even if 0 detections
    for cid, cname in names_dict.items():
        if cname not in per_class:
            per_class[cname] = {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }

    preprocess_ms = _safe_float(speed.get("preprocess", 0.0))
    inference_ms = _safe_float(speed.get("inference", 0.0))
    loss_ms = _safe_float(speed.get("loss", 0.0))
    postprocess_ms = _safe_float(speed.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + loss_ms + postprocess_ms
    fps = round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0

    return {
        "split": split_name,
        "overall": {
            "precision": round(mp, 5),
            "recall": round(mr, 5),
            "map50": round(map50, 5),
            "map50_95": round(map50_95, 5),
            "fitness": round(fitness, 5),
        },
        "per_class": per_class,
        "size_ap": getattr(val_results, "size_ap", {}),
        "speed_ms": {
            "preprocess": round(preprocess_ms, 3),
            "inference": round(inference_ms, 3),
            "loss": round(loss_ms, 3),
            "postprocess": round(postprocess_ms, 3),
            "total": round(total_ms, 3),
            "fps": fps,
        },
    }


def save_metrics_csv(metrics_list: List[Dict[str, Any]], csv_path: Union[str, Path]) -> None:
    """Save structured evaluation metrics list to CSV."""
    p = Path(csv_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in metrics_list:
        split = entry.get("split", "unknown")
        overall = entry.get("overall", {})
        speed = entry.get("speed_ms", {})

        # Overall row
        rows.append({
            "split": split,
            "category": "all_classes",
            "precision": overall.get("precision", 0.0),
            "recall": overall.get("recall", 0.0),
            "map50": overall.get("map50", 0.0),
            "map50_95": overall.get("map50_95", 0.0),
            "fitness": overall.get("fitness", 0.0),
            "inference_ms": speed.get("inference", 0.0),
            "fps": speed.get("fps", 0.0),
        })
        # Per class rows
        for cname, cstats in entry.get("per_class", {}).items():
            rows.append({
                "split": split,
                "category": cname,
                "precision": cstats.get("precision", 0.0),
                "recall": cstats.get("recall", 0.0),
                "map50": cstats.get("map50", 0.0),
                "map50_95": cstats.get("map50_95", 0.0),
                "fitness": "",
                "inference_ms": speed.get("inference", 0.0),
                "fps": speed.get("fps", 0.0),
            })

        for group, report in entry.get("size_ap", {}).items():
            for category, stats in [
                ("all_classes", report["overall"]),
                *report["per_class"].items(),
            ]:
                rows.append({
                    "split": split,
                    "category": f"{group}/{category}",
                    "precision": "",
                    "recall": "",
                    "map50": stats["map50"],
                    "map50_95": stats["map50_95"],
                    "fitness": "",
                    "inference_ms": "",
                    "fps": "",
                })

    fieldnames = [
        "split",
        "category",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "fitness",
        "inference_ms",
        "fps",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_yolo(
    weights: Optional[Union[str, Path]] = None,
    data: Optional[Union[str, Path]] = None,
    split: str = "test",
    batch: int = 16,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.6,
    device: Optional[Union[str, int]] = None,
    save_json: Optional[Union[str, Path]] = None,
    save_csv: Optional[Union[str, Path]] = None,
    project: str = "runs/val",
    name: str = "eval_test",
    verbose: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on validation or test split and export quantitative reports.

    Args:
        out_dir: Optional path identifying the evaluation run. Its final path
            component is used as the Ultralytics run name (for example,
            ``runs/val/yolo26n`` uses ``yolo26n``). It does not replace
            ``project``.
        project: Root directory for evaluation artifacts and metric reports.
            Defaults to ``runs/val``.

    Returns:
        Dict containing structured evaluation metrics for requested splits.
    """
    weights = weights if weights is not None else get_configured_weights_path()
    if weights is None:
        raise ValueError("Evaluation weights are not configured; set training.weights in the config.yaml")
    weights_path = Path(weights).resolve()
    data = data if data is not None else get_configured_data_yaml_path()
    if data is None:
        raise ValueError("Dataset path is not configured; set data_path in the config.yaml")
    data_path = Path(data).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights checkpoint not found: {weights_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML config not found: {data_path}")

    # Validate parameters
    if conf < 0.0 or conf > 1.0:
        raise ValueError(f"Confidence threshold conf must be between 0.0 and 1.0, got {conf}")
    if iou < 0.0 or iou > 1.0:
        raise ValueError(f"IoU threshold iou must be between 0.0 and 1.0, got {iou}")
    if batch < 1:
        raise ValueError(f"batch must be a positive integer, got {batch}")
    if imgsz < 32:
        raise ValueError(f"imgsz must be at least 32, got {imgsz}")
    if split not in ("val", "test", "train", "both", "all"):
        raise ValueError(f"Invalid split '{split}'. Allowed values: 'val', 'test', 'train', 'both', 'all'")

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    device_str = str(device)

    out_dir_path: Optional[Path] = None
    eval_name = name
    if out_dir is not None:
        out_dir_path = Path(out_dir).expanduser().resolve()
        if out_dir_path.name:
            eval_name = out_dir_path.name

    project_path = Path(project).expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)
    if save_json is None:
        save_json = project_path / eval_name / "metrics.json"
    if save_csv is None:
        save_csv = project_path / eval_name / "metrics.csv"

    logger.info(f"Loading model for evaluation: {weights_path}")
    model = YOLO(str(weights_path))

    splits_to_eval = ["val", "test"] if split in ("both", "all") else [split]
    project_path_str = str(project_path)
    results_payload: Dict[str, Any] = {
        "model": str(weights_path),
        "data_config": str(data_path),
        "device": device_str,
        "splits_evaluated": splits_to_eval,
        "results": {},
    }

    metrics_list_for_csv: List[Dict[str, Any]] = []

    for s_name in splits_to_eval:
        logger.info(f"Running evaluation on split: '{s_name}'...")
        val_res = model.val(
            validator=SizeAwareDetectionValidator,
            data=str(data_path),
            split=s_name,
            batch=batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device_str,
            project=project_path_str,
            name=f"{eval_name}",
            save_json=False,
            verbose=verbose,
        )

        extracted = extract_metrics_dict(val_res, s_name)
        results_payload["results"][s_name] = extracted
        metrics_list_for_csv.append(extracted)
        logger.info(
            f"Split [{s_name}] -> Precision: {extracted['overall']['precision']}, "
            f"Recall: {extracted['overall']['recall']}, "
            f"mAP50: {extracted['overall']['map50']}, "
            f"mAP50-95: {extracted['overall']['map50_95']}"
        )
        for group, report in extracted["size_ap"].items():
            logger.info(
                f"Split [{s_name}] {group.title()} AP -> "
                f"mAP50: {report['overall']['map50']}, "
                f"mAP50-95: {report['overall']['map50_95']}"
            )

    if save_json:
        json_p = Path(save_json).resolve()
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2)
        logger.info(f"Saved evaluation JSON report to: {json_p}")

    if save_csv:
        csv_p = Path(save_csv).resolve()
        save_metrics_csv(metrics_list_for_csv, csv_p)
        logger.info(f"Saved evaluation CSV report to: {csv_p}")

    return results_payload


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for model evaluation."""
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Compute detection metrics and Small/Tiny AP on dataset splits",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights (default: training.weights from config.yaml)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML config (default: data_path from config.yaml)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train", "both", "all"], help="Split to evaluate (default: test)")
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", "--img-size", dest="imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold (default: 0.6)")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (default: auto)")
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default=None,
        help="Path whose final component is used as the evaluation run name (does not replace project)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Path to save output JSON metrics (defaults to <project>/metrics.json)",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Path to save output CSV metrics (defaults to <project>/metrics.csv)",
    )
    parser.add_argument("--project", type=str, default="runs/val", help="Evaluation project directory (default: runs/val)")
    parser.add_argument("--name", type=str, default="eval_test", help="Evaluation run name (default: eval_test)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        evaluate_yolo(
            weights=args.weights,
            data=args.data,
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_json=args.save_json,
            save_csv=args.save_csv,
            project=args.project,
            name=args.name,
            out_dir=args.out_dir,
        )
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
