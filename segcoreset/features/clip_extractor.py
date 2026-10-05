"""CLIP ViT global embedding — optional extra representation point (§6 `clip_cls`)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .base import FeatureExtractor


class ClipExtractor(FeatureExtractor):
    def __init__(self, cfg):
        """Load the CLIP image processor and vision tower (with projection) from the HF Hub.

        Args:
            cfg: The resolved `features:` sub-config (needs `model_name`, e.g.
                `openai/clip-vit-base-patch32`).
        """
        super().__init__(cfg)
        self.outputs = ["cls"]
        self.processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(cfg.model_name)
        self.model.eval().to(self.device)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply CLIP's own resize/center-crop/normalize via `CLIPImageProcessor`.

        Args:
            image: PIL image.

        Returns:
            A `[3, H, W]` float tensor at CLIP's input size.
        """
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def extract_batch(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the projected CLIP image embedding as `{"cls": [B, D]}`, L2-normalized, on CPU.

        Args:
            batch: `[B, 3, H, W]` stack of `preprocess` outputs.
        """
        out = self.model(pixel_values=batch.to(self.device))
        return {"cls": F.normalize(out.image_embeds, dim=-1).cpu()}
