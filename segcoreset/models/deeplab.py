"""DeepLabV3+ / ResNet-50 (§3 Stage-4 CNN generalization check) via `segmentation_models_pytorch`."""
from __future__ import annotations

import segmentation_models_pytorch as smp

from .base import SegmentationModel


class DeepLabV3PlusWrapper(SegmentationModel):
    def __init__(self, model_cfg, num_classes: int):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=model_cfg.encoder_name,
            encoder_weights=model_cfg.encoder_weights,
            classes=num_classes,
        )

    def forward(self, pixel_values):
        return self.model(pixel_values)
