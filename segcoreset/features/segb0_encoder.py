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

# Prefixes in front of the bare `SegformerModel` keys, most specific first:
# `03_train.py`'s `SegFormerWrapper` (`model.segformer.`) and a raw HF
# `SegformerForSemanticSegmentation` state dict (`segformer.`).
_ENCODER_PREFIXES = ("model.segformer.", "segformer.")


def _load_encoder_state(path) -> dict:
    """Read a checkpoint and return just the encoder weights, keyed for `SegformerModel`.

    Accepts a `03_train.py` checkpoint (`{"model": state_dict, "optimizer": ..., ...}`), a
    bare `SegFormerWrapper` / `SegformerForSemanticSegmentation` state dict, or a bare
    `SegformerModel` state dict. Decode-head weights are dropped. The caller loads the
    result with `strict=True`, so a wrong file fails loudly instead of silently leaving the
    ImageNet weights in place.

    Args:
        path: Checkpoint file.

    Returns:
        State dict with `SegformerModel` key names.
    """
    state = torch.load(path, map_location="cpu")
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    for prefix in _ENCODER_PREFIXES:
        if any(k.startswith(prefix) for k in state):
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return state


class SegFormerEncoderExtractor(FeatureExtractor):
    """Uses the ImageNet-pretrained encoder by default; point `cfg.checkpoint` at a short
    full-data-trained run's weights to use a segmentation-adapted encoder instead."""

    def __init__(self, cfg):
        """Load the SegFormer encoder, optionally overriding its weights from a checkpoint.

        Args:
            cfg: The resolved `features:` sub-config (needs `pretrained` and `image_size`;
                optional `checkpoint`, see `_load_encoder_state`).
        """
        super().__init__(cfg)
        self.outputs = ["cls"]
        self.model = SegformerModel.from_pretrained(cfg.pretrained)
        if cfg.get("checkpoint"):
            self.model.load_state_dict(_load_encoder_state(cfg.checkpoint), strict=True)
        self.model.eval().to(self.device)
        self._transform = T.Compose([
            T.Resize((cfg.image_size, cfg.image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Bilinear-resize to a square `image_size`, convert to tensor, ImageNet-normalize.

        Args:
            image: PIL image.

        Returns:
            `[3, image_size, image_size]` float tensor.
        """
        return self._transform(image.convert("RGB"))

    @torch.no_grad()
    def extract_batch(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Global-average-pool the last encoder stage into `{"cls": [B, D]}`, L2-normalized, on CPU.

        Args:
            batch: `[B, 3, H, W]` stack of `preprocess` outputs.
        """
        out = self.model(pixel_values=batch.to(self.device))
        feat = out.last_hidden_state.mean(dim=(2, 3))  # global average pool over the last stage
        return {"cls": F.normalize(feat, dim=-1).cpu()}
