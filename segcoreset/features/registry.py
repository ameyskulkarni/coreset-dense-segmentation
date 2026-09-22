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
    backend = features_cfg.backend
    if backend not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature backend '{backend}'. Available: {list(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[backend](features_cfg)
