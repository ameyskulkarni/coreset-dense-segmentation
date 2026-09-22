"""Feature cache (§6, §12.5: "cache features once, never recompute"). One `.pt` file per
image under `root/<image_id>.pt`, plus an `index.json` manifest. Extraction is resumable —
already-cached ids are skipped on the next run."""
from __future__ import annotations

import json
from pathlib import Path

import torch


class FeatureStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, image_id: str) -> Path:
        return self.root / f"{image_id}.pt"

    def exists(self, image_id: str) -> bool:
        return self.path_for(image_id).exists()

    def save(self, image_id: str, features: dict[str, torch.Tensor]) -> None:
        torch.save(features, self.path_for(image_id))

    def load(self, image_id: str) -> dict[str, torch.Tensor]:
        return torch.load(self.path_for(image_id), map_location="cpu")

    def write_index(self, image_ids: list[str], meta: dict) -> None:
        with open(self.root / "index.json", "w") as f:
            json.dump({"image_ids": image_ids, **meta}, f, indent=2)

    def cached_ids(self) -> set[str]:
        return {p.stem for p in self.root.glob("*.pt")}
