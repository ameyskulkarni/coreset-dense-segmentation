"""DINOv2 (torch.hub) / DINOv3 (HuggingFace transformers) ViT features — the representation
the rest of the plan is built on (§6 `dinov2_cls` / `dinov2_patch`): the CLS token feeds
the global-embedding baselines (prototypicality, k-center, SemDeDup, ZCore), and the L2-
normalized patch-token grid feeds `patch_coverage` (§5.3).

Both weight sets download automatically on first use (torch.hub cache / HF hub cache).
DINOv2 is the plan's tested default; DINOv3 is provided as a drop-in config swap but its
HF output format (register-token count, key names) should be spot-checked against your
installed `transformers` version before trusting numbers from that path.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .base import FeatureExtractor

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoExtractor(FeatureExtractor):
    def __init__(self, cfg):
        """Load a DINOv2 (torch.hub) or DINOv3 (HF transformers) ViT and its preprocessing.

        Args:
            cfg: The resolved `features:` sub-config. Needs `source` (`torch_hub` or `hf`),
                `model_name`, `patch_size`, `image_size`, and `hub_repo` for torch.hub;
                optional `outputs` (default `["cls", "patch"]`).

        Raises:
            ValueError: If `image_size` is not a multiple of `patch_size`, or `source` is
                unknown.
        """
        super().__init__(cfg)
        self.outputs = list(cfg.get("outputs", ["cls", "patch"]))
        self.patch_size = cfg.patch_size
        self.image_size = cfg.image_size
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size ({self.image_size}) must be a multiple of patch_size ({self.patch_size})")
        self._grid = self.image_size // self.patch_size
        self._source = cfg.source

        if self._source == "torch_hub":
            self.model = torch.hub.load(cfg.hub_repo, cfg.model_name, trust_repo=True)
        elif self._source == "hf":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(cfg.model_name)
        else:
            raise ValueError(f"Unknown DINO source '{cfg.source}'")
        self.model.eval().to(self.device)

        self._transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Bicubic-resize to a square `image_size`, convert to tensor, ImageNet-normalize.

        The square resize ignores aspect ratio so every image yields the same patch grid.

        Args:
            image: PIL image.

        Returns:
            `[3, image_size, image_size]` float tensor.
        """
        return self._transform(image.convert("RGB"))

    @torch.no_grad()
    def extract_batch(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract the normalized CLS token and/or the patch-token grid.

        For torch.hub DINOv2 the final-layer-normed tokens (`x_norm_clstoken`,
        `x_norm_patchtokens`) are used. For HF DINOv3 the CLS token is position 0 of
        `last_hidden_state` and the patch tokens follow the register tokens. Patch tokens are
        reshaped to a square grid (falling back to the configured grid if the token count is not
        a perfect square).

        Args:
            batch: `[B, 3, image_size, image_size]` stack of `preprocess` outputs.

        Returns:
            A dict restricted to `self.outputs`: `"cls"` `[B, D]` and/or `"patch"`
            `[B, grid, grid, D]`, each L2-normalized along D, on CPU.
        """
        batch = batch.to(self.device)
        if self._source == "torch_hub":
            feats = self.model.forward_features(batch)
            cls, patch = feats["x_norm_clstoken"], feats["x_norm_patchtokens"]
        else:  # HF DINOv3
            hf_out = self.model(pixel_values=batch)
            tokens = hf_out.last_hidden_state
            num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
            cls, patch = tokens[:, 0], tokens[:, 1 + num_register_tokens :]

        b, n, d = patch.shape
        grid = int(round(n ** 0.5))
        if grid * grid != n:
            grid = self._grid  # fall back to the configured grid if tokens were padded
        patch = patch[:, : grid * grid].reshape(b, grid, grid, d)

        out = {}
        if "cls" in self.outputs:
            out["cls"] = F.normalize(cls, dim=-1).cpu()
        if "patch" in self.outputs:
            out["patch"] = F.normalize(patch, dim=-1).cpu()
        return out
