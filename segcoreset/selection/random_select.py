"""The floor baseline (§5.1): uniform random subset, no features used."""
from __future__ import annotations

import random as _random
from typing import Sequence


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    """Pick `budget` ids uniformly at random without replacement.

    Uses a private `random.Random(seed)`, so the result depends only on `seed` and the
    order of `image_ids` (not on global RNG state).

    Args:
        image_ids: Candidate pool.
        features: Ignored.
        budget: Number of ids to keep.
        seed: RNG seed.
        cfg: Ignored.

    Returns:
        The chosen ids, sorted.
    """
    rng = _random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    return sorted(ids[:budget])
