"""Model registry — swap architectures from config alone (`model: segformer_b0` etc.)."""
from __future__ import annotations

from .deeplab import DeepLabV3PlusWrapper
from .segformer import SegFormerWrapper
from .unet import UNetWrapper

MODEL_REGISTRY = {
    "segformer": SegFormerWrapper,
    "deeplabv3plus": DeepLabV3PlusWrapper,
    "unet": UNetWrapper,
}


def build_model(model_cfg, num_classes: int):
    """Instantiate the architecture registered under `model_cfg.arch`.

    Args:
        model_cfg: The resolved `model:` sub-config.
        num_classes: Number of output classes (from the dataset config).

    Returns:
        A `SegmentationModel` (not yet moved to any device).

    Raises:
        ValueError: If `model_cfg.arch` is not in `MODEL_REGISTRY`.
    """
    arch = model_cfg.arch
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model arch '{arch}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](model_cfg, num_classes)
