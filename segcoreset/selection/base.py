"""Common selector contract (§12.2). Every method is one module exposing a `select`
function with this signature, registered in `registry.py` — adding a new label-free
baseline (prototypicality, k-center, SemDeDup, ZCore, `patch_coverage`, ...) never touches
the trainer or eval code."""
from __future__ import annotations

from typing import Sequence


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    """Return `budget` chosen ids from `image_ids`, deterministic given `seed`. No labels.

    `features` is whatever the concrete method needs — e.g. a `FeatureStore` to load
    cached DINOv2 embeddings from — or `None` for methods that don't need any (`random`,
    `full`). `cfg` is the resolved `selection:` sub-config for this method.
    """
    raise NotImplementedError
