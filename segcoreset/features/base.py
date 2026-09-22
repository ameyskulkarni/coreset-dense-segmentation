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
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> a single model-input tensor (already resized/normalized)."""
        raise NotImplementedError

    @torch.no_grad()
    def extract_batch(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """`batch`: [B, C, H, W]. Returns `{"cls": [B, D]}` and/or `{"patch": [B, Hp, Wp, D]}`,
        L2-normalized, on CPU."""
        raise NotImplementedError
