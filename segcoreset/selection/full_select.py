"""Reference denominator (§4): the entire train set. `budget` is ignored (kept for
interface uniformity) — used for the matched-budget and long-budget ceiling runs, not as
a competing selection method."""
from __future__ import annotations

from typing import Sequence


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    return sorted(image_ids)
