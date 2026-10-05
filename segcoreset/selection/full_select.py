"""Reference denominator (§4): the entire train set. `budget` is ignored (kept for
interface uniformity) — used for the matched-budget and long-budget ceiling runs, not as
a competing selection method."""
from __future__ import annotations

from typing import Sequence


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    """Return every candidate id, sorted. `features`, `budget`, `seed`, and `cfg` are ignored.

    Args:
        image_ids: Candidate pool.
        features: Ignored.
        budget: Ignored.
        seed: Ignored.
        cfg: Ignored.

    Returns:
        `sorted(image_ids)`.
    """
    return sorted(image_ids)
