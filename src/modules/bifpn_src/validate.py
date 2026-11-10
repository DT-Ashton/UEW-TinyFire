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


def validate_attention_modulation() -> None:
    """Check identity initialization and both signs of the spatial residual."""
    attention = SpatialRelationalCoAttentionLite(channels=8, pool_size=2)
    feature = torch.ones(1, 8, 4, 4)
    with torch.no_grad():
        attention.spatial.weight.zero_()
        attention.qkv.weight.zero_()
        attention.relation_spatial.weight.zero_()
        attention.relation_channel.weight.zero_()
        attention.spatial.bias.fill_(2.0)
        assert torch.equal(attention(feature), feature), "Zero gain must preserve the feature exactly"

        attention.gain.fill_(1.0)
        enhanced = attention(feature)
        attention.spatial.bias.fill_(-2.0)
        suppressed = attention(feature)
    assert torch.all(enhanced > feature), "Positive spatial evidence must enhance the feature"
    assert torch.all(suppressed < feature), "Negative spatial evidence must suppress the feature"


def validate(imgsz: int = 256) -> None:
    """Build the YAML and check its feature graph and stock training loss."""
    if imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz must be at least 32 and divisible by 32")

    torch.manual_seed(0)
    torch.set_num_threads(2)
    network = YOLO(str(ROOT / "yolo26n-p2_bifpn_src.yaml"), verbose=False).model
    assert len(network.yaml["backbone"]) == 11, "Expected the 11-layer YOLO26n backbone"
    assert not any(
        isinstance(module, SpatialRelationalCoAttentionLite) for module in network.model[:11]
    ), "SRC must not appear in the backbone"
    attention_layers = [
        index for index, module in enumerate(network.model)
        if isinstance(module, SpatialRelationalCoAttentionLite)
    ]
    assert attention_layers == [30, 41], f"Expected SRC only after neck fusion at 30/41, got {attention_layers}"
    fusion_layers = [
        index for index, module in enumerate(network.model)
        if isinstance(module, WeightedBiFPNFusion)
    ]
    assert fusion_layers == [17, 22, 27, 33, 38, 44], f"Expected six BiFPN fusions, got {fusion_layers}"
    head = network.model[-1]
    assert isinstance(head, Detect), f"Expected stock Detect head, got {type(head).__name__}"
    assert head.nl == 4 and len(head.f) == 4, f"Expected four detection levels, got {head.nl}"
    assert head.f == [30, 35, 41, 46], f"Unexpected Detect inputs: {head.f}"
    strides = [int(value) for value in network.stride.tolist()]
    assert strides == [4, 8, 16, 32], f"Unexpected strides: {strides}"
    validate_attention_modulation()

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
    assert len(fusions) == 6 and len(attentions) == 2, "Unexpected fusion or attention node count"
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

    print(f"Backbone: 11 unchanged layers; neck SRC {attention_layers}; BiFPN fusions {fusion_layers}")
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
