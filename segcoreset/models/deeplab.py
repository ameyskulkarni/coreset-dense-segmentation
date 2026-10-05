"""DeepLabV3+ / ResNet-50 (§3 Stage-4 CNN generalization check) via `segmentation_models_pytorch`."""
from __future__ import annotations

import segmentation_models_pytorch as smp

from .base import SegmentationModel


class DeepLabV3PlusWrapper(SegmentationModel):
    def __init__(self, model_cfg, num_classes: int):
        """Build an `smp.DeepLabV3Plus` with a pretrained encoder and a fresh head.

        Args:
            model_cfg: The resolved `model:` sub-config (needs `encoder_name`, e.g.
                `resnet50`, and `encoder_weights`, e.g. `imagenet` or `None`).
            num_classes: Number of output classes.
        """
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=model_cfg.encoder_name,
            encoder_weights=model_cfg.encoder_weights,
            classes=num_classes,
        )

    def forward(self, pixel_values):
        """Return logits `[B, num_classes, H, W]` (smp already upsamples to input size).

        Args:
            pixel_values: Normalized images, `[B, 3, H, W]`; smp raises if H/W are not
                divisible by the encoder's output stride.
        """
        return self.model(pixel_values)
