"""Rare-class computation (§2): pixel-frequency ranking over the FULL train set, computed
ONCE per dataset and frozen to disk. Labels are used here for analysis only, never for
selection; every downstream eval reports rare-class mIoU against this frozen list.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .registry import build_dataset


def compute_and_cache_rare_classes(dataset_cfg, k: int, cache_path: Path | str | None = None) -> dict:
    """Rank classes by labeled-pixel count over the FULL train split and freeze the rarest `k`.

    Reads label PNGs directly (no image loading or transforms), counts pixels per encoded
    class (ignoring `ignore_index`), and writes the result as JSON. Labels are used here for
    analysis only — the output feeds rare-class mIoU in evaluation, never selection.

    Args:
        dataset_cfg: The resolved `dataset:` sub-config (needs `name`, `train_split`, and
            `rare_classes_cache` unless `cache_path` is given).
        k: Number of rarest classes to keep.
        cache_path: Output JSON path; defaults to `dataset_cfg.rare_classes_cache`.

    Returns:
        The dict written to disk: `dataset`, `k`, `rare_class_ids` (rarest first),
        `rare_class_names` (or `None` if the dataset has no `class_names`), and per-class
        `pixel_counts` / `pixel_freq` (indexed by class id).
    """
    ds = build_dataset(dataset_cfg, split=dataset_cfg.train_split)
    counts = np.zeros(ds.num_classes, dtype=np.int64)
    for s in tqdm(ds.samples, desc=f"counting pixels [{dataset_cfg.name}]"):
        raw = np.array(Image.open(s.label_path))
        label = ds._encode_label(raw)
        valid = label != ds.ignore_index
        counts += np.bincount(label[valid].ravel(), minlength=ds.num_classes)[: ds.num_classes]

    freq = counts / counts.sum()
    order = np.argsort(counts)  # ascending: rarest first
    rare_ids = order[:k].tolist()
    class_names = getattr(ds, "class_names", None)

    result = {
        "dataset": dataset_cfg.name,
        "k": k,
        "rare_class_ids": rare_ids,
        "rare_class_names": [class_names[i] for i in rare_ids] if class_names else None,
        "pixel_counts": counts.tolist(),
        "pixel_freq": freq.tolist(),
    }
    cache_path = Path(cache_path or dataset_cfg.rare_classes_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_rare_classes(cache_path: Path | str) -> list[int]:
    """Load the frozen rare-class ids written by `compute_and_cache_rare_classes`.

    Args:
        cache_path: Path to `results/rare_classes/<dataset>.json`.

    Returns:
        The rare class ids, rarest first.
    """
    with open(cache_path) as f:
        return json.load(f)["rare_class_ids"]
