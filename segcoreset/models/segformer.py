"""SegFormer-B0 (§3 primary workhorse) via HuggingFace `transformers`."""
from __future__ import annotations

import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .base import SegmentationModel


class SegFormerWrapper(SegmentationModel):
    """The HF decode head outputs logits at 1/4 input resolution; bilinear-upsampled to
    full resolution here so every model in the registry shares the same output contract.
    `model_cfg.pretrained` (e.g. `nvidia/mit-b0`) is the ImageNet-1k-pretrained ENCODER —
    the decode head is randomly initialized and trained from scratch every run."""

    def __init__(self, model_cfg, num_classes: int):
        """Load the pretrained SegFormer encoder and attach a fresh `num_classes` decode head.

        `ignore_mismatched_sizes=True` lets the classifier be re-created at the requested size
        without erroring when the checkpoint's head shape differs (or is absent).

        Args:
            model_cfg: The resolved `model:` sub-config (needs `pretrained`, a HF Hub id or
                local path such as `nvidia/mit-b0`).
            num_classes: Number of output classes.
        """
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_cfg.pretrained, num_labels=num_classes, ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values):
        """Run SegFormer and bilinearly upsample its 1/4-resolution logits to input size.

        Args:
            pixel_values: Normalized images, `[B, 3, H, W]`.

        Returns:
            Logits `[B, num_classes, H, W]`.
        """
        logits = self.model(pixel_values=pixel_values).logits
        return F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
