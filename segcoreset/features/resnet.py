"""ImageNet-supervised ResNet-50 — the classification-representation arm of the
representation study (§6 `rn50_sup`, Contribution 3)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

from .base import FeatureExtractor

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetExtractor(FeatureExtractor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.outputs = ["cls"]
        weights = tvm.get_model_weights(cfg.model_name)[cfg.get("weights", "IMAGENET1K_V2")]
        backbone = tvm.get_model(cfg.model_name, weights=weights)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])  # drop the fc head
        self.model.eval().to(self.device)
        self._transform = T.Compose([
            T.Resize(cfg.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transform(image.convert("RGB"))

    @torch.no_grad()
    def extract_batch(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.model(batch.to(self.device)).flatten(1)
        return {"cls": F.normalize(feats, dim=-1).cpu()}
