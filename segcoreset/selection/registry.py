"""Selector registry (§12.2). Only `random`/`full` are implemented today; the label-free
baselines (prototypicality, k-center, SemDeDup, bpp, ZCore) and `patch_coverage` (§5) drop
in as new modules here without touching the trainer, evaluator, or scripts."""
from __future__ import annotations

from . import full_select, random_select

SELECTOR_REGISTRY = {
    "random": random_select.select,
    "full": full_select.select,
}


def get_selector(method: str):
    if method not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selection method '{method}'. Available: {list(SELECTOR_REGISTRY)}. "
            "Add new methods as modules under segcoreset/selection/ — see base.py for the interface."
        )
    return SELECTOR_REGISTRY[method]
