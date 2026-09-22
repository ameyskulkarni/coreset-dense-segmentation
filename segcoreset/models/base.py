"""Common contract every architecture implements: `forward(pixel_values) -> logits` at the
ORIGINAL input resolution, `[B, num_classes, H, W]`. The trainer/evaluator never need to
know which backbone produced the logits.
"""
from __future__ import annotations

import torch.nn as nn


class SegmentationModel(nn.Module):
    def forward(self, pixel_values):
        raise NotImplementedError
