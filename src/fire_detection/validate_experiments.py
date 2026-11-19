"""Validate the three YOLO26n-P2 architecture experiments on CPU.

Run ``python src/fire_detection/validate_experiments.py`` from any directory.
No dataset, checkpoint, or network access is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils import IterableSimpleNamespace, YAML


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.bifpn_src import WeightedBiFPNFusion, register_modules as register_bifpn  # noqa: E402
from src.modules.co_attention import NeckSpatialRelationCoAttention, register_modules as register_src  # noqa: E402


EXPERIMENTS = (
    ("bifpn", "yolo26n-p2_bifpn.yaml", 6, 0, (64, 64, 64, 64)),
    ("src", "yolo26n-p2_src.yaml", 0, 2, (32, 64, 128, 256)),
    ("bifpn_src", "yolo26n-p2_bifpn_src.yaml", 6, 2, (64, 64, 64, 64)),
)


def _normalized_graph(layers: list) -> list:
    """Treat each inline SRC as its predecessor when comparing BiFPN routing."""
    mapping = {-1: -1}
    normalized = []
    for index, (sources, repeats, module, args) in enumerate(layers):
        def resolved(source: int) -> int:
            original = index + source if source < 0 else source
            return mapping[original]

        if module == "NeckSpatialRelationCoAttention":
            assert isinstance(sources, int), "An inline SRC must have one input"
            mapping[index] = resolved(sources)
            continue

        mapped_sources = (
            tuple(resolved(source) for source in sources)
            if isinstance(sources, list)
            else resolved(sources)
        )
        mapping[index] = len(normalized)
        normalized.append((mapped_sources, repeats, module, args))
    return normalized


def validate(imgsz: int = 256) -> None:
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be at least 32 and divisible by 32")

    register_bifpn()
    register_src()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    stock = YAML.load(Path(ultralytics.__file__).parent / "cfg/models/26/yolo26-p2.yaml")
    stock_graph = _normalized_graph(stock["backbone"] + stock["head"])
    bifpn_graph = None

    for name, filename, fusion_count, src_count, channels in EXPERIMENTS:
        config = YAML.load(ROOT / filename)
        assert config["backbone"] == stock["backbone"], f"{name}: backbone differs from stock YOLO26n-P2"
        assert len(config["backbone"]) == 11, f"{name}: expected 11 backbone layers"

        graph = _normalized_graph(config["backbone"] + config["head"])
        if name == "bifpn":
            bifpn_graph = graph
        elif name == "src":
            assert graph == stock_graph, "SRC-only PAN topology differs from stock YOLO26n-P2"
        elif name == "bifpn_src":
            assert graph == bifpn_graph, "BiFPN topology changed after inserting SRC"

        network = YOLO(str(ROOT / filename), verbose=False).model
        fusion_layers = [
            index for index, layer in enumerate(network.model) if isinstance(layer, WeightedBiFPNFusion)
        ]
        src_layers = [
            index for index, layer in enumerate(network.model) if isinstance(layer, NeckSpatialRelationCoAttention)
        ]
        assert len(fusion_layers) == fusion_count, f"{name}: unexpected BiFPN fusion count"
        assert len(src_layers) == src_count, f"{name}: unexpected neck SRC count"
        assert all(index >= 11 for index in src_layers), f"{name}: SRC appeared in the backbone"
        assert not any(
            isinstance(layer, NeckSpatialRelationCoAttention) for layer in network.model[:11]
        ), f"{name}: backbone contains SRC"

        head = network.model[-1]
        assert isinstance(head, Detect) and head.nl == 4, f"{name}: expected stock four-level Detect"
        strides = [int(stride) for stride in network.stride.tolist()]
        assert strides == [4, 8, 16, 32], f"{name}: unexpected strides {strides}"

        shapes = []

        def capture_features(_module, inputs) -> None:
            shapes.extend(tuple(feature.shape) for feature in inputs[0])

        hook = head.register_forward_pre_hook(capture_features)
        network.train()
        network.args = IterableSimpleNamespace(**network.args)
        batch = {
            "img": torch.rand(1, 3, imgsz, imgsz),
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.125, 0.125]]),
        }
        try:
            loss, _items = network.loss(batch)
        finally:
            hook.remove()

        expected_shapes = [
            (1, width, imgsz // stride, imgsz // stride)
            for width, stride in zip(channels, strides)
        ]
        assert shapes == expected_shapes, f"{name}: feature shapes {shapes} != {expected_shapes}"
        assert torch.isfinite(loss).all(), f"{name}: non-finite stock loss"
        loss.sum().backward()
        assert all(
            network.model[index].weights.grad is not None
            and torch.isfinite(network.model[index].weights.grad).all()
            and network.model[index].weights.grad.abs().sum() > 0
            for index in fusion_layers
        ), f"{name}: missing fusion weight gradients"
        assert all(
            network.model[index].gamma.grad is not None
            and torch.isfinite(network.model[index].gamma.grad).all()
            and network.model[index].gamma.grad.abs().sum() > 0
            for index in src_layers
        ), f"{name}: missing SRC residual gradients"
        print(
            f"{name}: strides={strides}, channels={list(channels)}, "
            f"fusion={len(fusion_layers)}, SRC={len(src_layers)}, loss={loss.sum().item():.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=256, help="Square input size divisible by 32")
    args = parser.parse_args()
    validate(args.imgsz)


if __name__ == "__main__":
    main()
