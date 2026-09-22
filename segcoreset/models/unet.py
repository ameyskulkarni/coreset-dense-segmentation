"""U-Net / ResNet-34 (§3) — fallback CNN if DeepLabV3+'s per-run time is too costly."""
from __future__ import annotations

import segmentation_models_pytorch as smp

from .base import SegmentationModel


class UNetWrapper(SegmentationModel):
    def __init__(self, model_cfg, num_classes: int):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=model_cfg.encoder_name,
            encoder_weights=model_cfg.encoder_weights,
            classes=num_classes,
        )

    def forward(self, pixel_values):
        return self.model(pixel_values)
