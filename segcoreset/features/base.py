"""Common interface for representation extractors (§6). Everything besides the model
forward pass (dataset iteration, caching, resuming) lives in `store.py` and
`scripts/00_extract_features.py`, so a new representation is one new subclass.
"""
from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image


class FeatureExtractor:
    name: str
    outputs: Sequence[str]  # subset of {"cls", "patch"}

    def __init__(self, cfg):
        """Store the config and pick the device (CUDA if available, else CPU).

        Subclasses call this first, then load their model onto `self.device` and set
        `self.outputs`.

        Args:
            cfg: The resolved `features:` sub-config for this representation.
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> a single model-input tensor (already resized/normalized).

        Runs inside DataLoader workers, so it must be CPU-only and picklable.

        Args:
            image: Any-mode PIL image (implementations convert to RGB).

        Returns:
            A `[C, H, W]` float tensor at the extractor's fixed input size.
        """
        raise NotImplementedError

    @torch.no_grad()
    def extract_batch(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the backbone on a preprocessed batch.

        Args:
            batch: `[B, C, H, W]` stack of `preprocess` outputs (any device).

        Returns:
            `{"cls": [B, D]}` and/or `{"patch": [B, Hp, Wp, D]}`, L2-normalized along D, on CPU.
        """
        raise NotImplementedError
