"""CPU smoke test for the custom YOLO26n BiFPN/SRC architecture.

Run from any directory with ``python src/modules/bifpn_src/validate.py``.
No checkpoint, dataset, or network access is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.bifpn_src import SpatialRelationalCoAttentionLite, WeightedBiFPNFusion  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import Detect  # noqa: E402
from ultralytics.utils import IterableSimpleNamespace  # noqa: E402


def validate(imgsz: int = 256) -> None:
    """Build the YAML and check its feature graph and stock training loss."""
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be at least 32 and divisible by 32")

    torch.manual_seed(0)
    torch.set_num_threads(2)
    network = YOLO(str(ROOT / "yolo26n_bifpn_src_p2.yaml"), verbose=False).model
    head = network.model[-1]
    assert isinstance(head, Detect), f"Expected stock Detect head, got {type(head).__name__}"
    assert head.nl == 4 and len(head.f) == 4, f"Expected four detection levels, got {head.nl}"
    strides = [int(value) for value in network.stride.tolist()]
    assert strides == [4, 8, 16, 32], f"Unexpected strides: {strides}"

    shapes: list[tuple[int, ...]] = []

    def capture_features(_module, inputs) -> None:
        shapes.extend(tuple(feature.shape) for feature in inputs[0])

    hook = head.register_forward_pre_hook(capture_features)
    network.train()
    # Ultralytics' trainer normally converts YAML arguments to this namespace
    # before creating the stock criterion; the standalone model keeps a dict.
    network.args = IterableSimpleNamespace(**network.args)
    batch = {
        "img": torch.rand(1, 3, imgsz, imgsz),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),  # class 0: fire
        "bboxes": torch.tensor([[0.5, 0.5, 0.125, 0.125]]),  # normalized xywh
    }
    try:
        loss, _items = network.loss(batch)
    finally:
        hook.remove()

    expected = [(1, 64, imgsz // stride, imgsz // stride) for stride in strides]
    assert shapes == expected, f"Unexpected P2-P5 feature shapes: {shapes}, expected {expected}"
    assert torch.isfinite(loss).all(), f"Non-finite stock loss: {loss}"
    loss.sum().backward()

    fusions = [module for module in network.modules() if isinstance(module, WeightedBiFPNFusion)]
    attentions = [module for module in network.modules() if isinstance(module, SpatialRelationalCoAttentionLite)]
    assert fusions and attentions, "Custom fusion or attention nodes are missing"
    assert all(
        module.weights.grad is not None
        and torch.isfinite(module.weights.grad).all()
        and module.weights.grad.abs().sum() > 0
        for module in fusions
    ), "A fusion weight has no finite, nonzero gradient"
    assert all(
        module.gain.grad is not None
        and torch.isfinite(module.gain.grad).all()
        and module.gain.grad.abs().sum() > 0
        for module in attentions
    ), "An attention residual gain has no finite, nonzero gradient"

    print(f"Detect: {head.nl} levels; strides {strides}")
    print(f"Features: {shapes}")
    print(f"Stock loss: {float(loss.sum().detach()):.4f}; gradients: {len(fusions)} fusions, {len(attentions)} attentions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=256, help="Square input size, divisible by 32")
    args = parser.parse_args()
    validate(args.imgsz)


if __name__ == "__main__":
    main()
