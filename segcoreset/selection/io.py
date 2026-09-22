"""Subset id-list persistence (§12.2): `results/subsets/{dataset}_{method}_{ratio}_{seed}.json`."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SUBSETS_DIR = Path("results/subsets")


def subset_path(method: str, dataset: str, ratio: float, seed: int, root: Path | str = DEFAULT_SUBSETS_DIR) -> Path:
    return Path(root) / f"{dataset}_{method}_{ratio:g}_{seed}.json"


def save_subset(path: Path, image_ids: list[str], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"image_ids": image_ids, **meta}, f, indent=2)


def load_subset(path: Path | str) -> list[str]:
    with open(path) as f:
        return json.load(f)["image_ids"]
