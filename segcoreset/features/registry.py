"""Representation registry (§6) — adding a representation is one new file + one line here."""
from __future__ import annotations

from .clip_extractor import ClipExtractor
from .dino import DinoExtractor
from .resnet import ResNetExtractor
from .segb0_encoder import SegFormerEncoderExtractor

FEATURE_REGISTRY = {
    "dinov2": DinoExtractor,
    "dinov3": DinoExtractor,
    "resnet": ResNetExtractor,
    "segb0_encoder": SegFormerEncoderExtractor,
    "clip": ClipExtractor,
}


def build_extractor(features_cfg):
    """Instantiate the extractor registered under `features_cfg.backend`.

    Constructing an extractor loads (and on first use downloads) its pretrained weights.

    Args:
        features_cfg: The resolved `features:` sub-config.

    Returns:
        A `FeatureExtractor` with its model on the selected device.

    Raises:
        ValueError: If `features_cfg.backend` is not in `FEATURE_REGISTRY`.
    """
    backend = features_cfg.backend
    if backend not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature backend '{backend}'. Available: {list(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[backend](features_cfg)
