"""Common contract every architecture implements: `forward(pixel_values) -> logits` at the
ORIGINAL input resolution, `[B, num_classes, H, W]`. The trainer/evaluator never need to
know which backbone produced the logits.
"""
from __future__ import annotations

import torch.nn as nn


class SegmentationModel(nn.Module):
    def forward(self, pixel_values):
        """Predict per-pixel class logits.

        Args:
            pixel_values: Normalized images, `[B, 3, H, W]`.

        Returns:
            Logits `[B, num_classes, H, W]` at the SAME spatial size as the input.
        """
        raise NotImplementedError
