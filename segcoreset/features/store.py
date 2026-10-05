"""Feature cache (§6, §12.5: "cache features once, never recompute"). One `.pt` file per
image under `root/<image_id>.pt`, plus an `index.json` manifest. Extraction is resumable —
already-cached ids are skipped on the next run."""
from __future__ import annotations

import json
from pathlib import Path

import torch


class FeatureStore:
    def __init__(self, root: Path | str):
        """Open (creating if needed) a cache directory of per-image feature files.

        Args:
            root: Directory, conventionally `results/features/<dataset>/<backbone>/<split>`.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, image_id: str) -> Path:
        """Return the `.pt` path for `image_id` (whether or not it exists)."""
        return self.root / f"{image_id}.pt"

    def exists(self, image_id: str) -> bool:
        """Return whether features for `image_id` are already cached."""
        return self.path_for(image_id).exists()

    def save(self, image_id: str, features: dict[str, torch.Tensor]) -> None:
        # .clone(): a slice `batch[i]` is a view, and torch.save writes the WHOLE underlying
        # storage — without this every file would hold the entire batch (32x bloat).
        """Write one image's feature tensors to `<root>/<image_id>.pt`, overwriting any existing file.

        Args:
            image_id: Image id (used as the filename stem).
            features: Name -> tensor (e.g. `{"cls": [D], "patch": [Hp, Wp, D]}`); each is
                detached and cloned so only its own data is serialized.
        """
        torch.save({k: v.detach().clone() for k, v in features.items()}, self.path_for(image_id))

    def load(self, image_id: str) -> dict[str, torch.Tensor]:
        """Load one image's cached feature dict onto the CPU.

        Args:
            image_id: Image id.

        Returns:
            The dict passed to `save`.

        Raises:
            FileNotFoundError: If the image is not cached.
        """
        return torch.load(self.path_for(image_id), map_location="cpu")

    def write_index(self, image_ids: list[str], meta: dict) -> None:
        """Write `<root>/index.json` describing the cache.

        Args:
            image_ids: All image ids in the split (stored under `"image_ids"`).
            meta: Extra top-level fields (e.g. `dataset`, `representation`, `split`,
                `image_paths`). `00b_derive_embeddings.py` relies on `image_paths`.
        """
        with open(self.root / "index.json", "w") as f:
            json.dump({"image_ids": image_ids, **meta}, f, indent=2)

    def cached_ids(self) -> set[str]:
        """Return the set of image ids that have a `.pt` file in the cache."""
        return {p.stem for p in self.root.glob("*.pt")}
