"""Subset id-list persistence (§12.2): `results/subsets/{dataset}_{method}_{ratio}_{seed}.json`."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SUBSETS_DIR = Path("results/subsets")


def subset_path(method: str, dataset: str, ratio: float, seed: int, root: Path | str = DEFAULT_SUBSETS_DIR) -> Path:
    """Return the canonical subset file path `<root>/<dataset>_<method>_<ratio:g>_<seed>.json`.

    `03_train.py` parses method and ratio back out of this filename by splitting on `_`,
    so `method` must not contain underscores.

    Args:
        method: Selection name (resolved `selection.name`, e.g. `random`, `proto-hard-k20-bal50-dinov2vitb14-cls`).
        dataset: Dataset name.
        ratio: Fraction of the train set kept (formatted with `:g`, e.g. `0.1`).
        seed: Selection seed.
        root: Subsets directory.

    Returns:
        The path (not created).
    """
    return Path(root) / f"{dataset}_{method}_{ratio:g}_{seed}.json"


def save_subset(path: Path, image_ids: list[str], meta: dict) -> None:
    """Write a subset file: `{"image_ids": [...], **meta}` as indented JSON.

    Args:
        path: Destination (parents created).
        image_ids: The selected ids.
        meta: Provenance fields stored alongside (dataset, method, ratio, seed, ...).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"image_ids": image_ids, **meta}, f, indent=2)


def load_subset(path: Path | str) -> list[str]:
    """Read just the `image_ids` list from a subset file.

    Args:
        path: Subset JSON path.

    Returns:
        The selected image ids, in file order.
    """
    with open(path) as f:
        return json.load(f)["image_ids"]
