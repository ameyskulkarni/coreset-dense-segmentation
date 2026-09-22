"""SegFormer-B0 encoder, globally pooled — the segmentation-representation arm of the
representation study (§6 `segb0_enc`, Contribution 3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import SegformerModel

from .base import FeatureExtractor

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SegFormerEncoderExtractor(FeatureExtractor):
    """Uses the ImageNet-pretrained encoder by default; point `cfg.checkpoint` at a short
    full-data-trained run's weights to use a segmentation-adapted encoder instead."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = ["cls"]
        self.model = SegformerModel.from_pretrained(cfg.pretrained)
        if cfg.get("checkpoint"):
            state = torch.load(cfg.checkpoint, map_location="cpu")
            self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)
        self._transform = T.Compose([
            T.Resize((cfg.image_size, cfg.image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transform(image.convert("RGB"))

    @torch.no_grad()
    def extract_batch(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(pixel_values=batch.to(self.device))
        feat = out.last_hidden_state.mean(dim=(2, 3))  # global average pool over the last stage
        return {"cls": F.normalize(feat, dim=-1).cpu()}
