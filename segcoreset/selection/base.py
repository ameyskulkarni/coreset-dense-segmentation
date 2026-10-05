"""Common selector contract (§12.2). Every method is one module exposing a `select`
function with this signature, registered in `registry.py` — adding a new label-free
baseline (prototypicality, k-center, SemDeDup, ZCore, `patch_coverage`, ...) never touches
the trainer or eval code."""
from __future__ import annotations

from typing import Sequence


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    """Return `budget` chosen ids from `image_ids`, deterministic given `seed`. No labels.

    `features` is whatever the concrete method needs — for feature-based methods,
    `02_select.py` passes an `[N, D]` embedding matrix row-aligned with `image_ids` (loaded
    from the selection config's `embedding:` block) — or `None` for methods that don't need
    any (`random`, `full`). `cfg` is the resolved `selection:` sub-config for this method.

    Args:
        image_ids: Candidate pool (the full train split's ids).
        features: Method-specific inputs, or `None`.
        budget: Number of ids to return.
        seed: Seed for any stochastic step.
        cfg: The resolved `selection:` sub-config.

    Returns:
        The chosen ids (implementations in this repo return them sorted).
    """
    raise NotImplementedError
