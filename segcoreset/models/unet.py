"""U-Net / ResNet-34 (§3) — fallback CNN if DeepLabV3+'s per-run time is too costly."""
from __future__ import annotations

import segmentation_models_pytorch as smp

from .base import SegmentationModel


class UNetWrapper(SegmentationModel):
    def __init__(self, model_cfg, num_classes: int):
        """Build an `smp.Unet` with a pretrained encoder and a fresh decoder/head.

        Args:
            model_cfg: The resolved `model:` sub-config (needs `encoder_name`, e.g.
                `resnet34`, and `encoder_weights`).
            num_classes: Number of output classes.
        """
        super().__init__()
        self.model = smp.Unet(
            encoder_name=model_cfg.encoder_name,
            encoder_weights=model_cfg.encoder_weights,
            classes=num_classes,
        )

    def forward(self, pixel_values):
        """Return logits `[B, num_classes, H, W]` at input resolution.

        Args:
            pixel_values: Normalized images, `[B, 3, H, W]`; H and W must be divisible by 32.
        """
        return self.model(pixel_values)
