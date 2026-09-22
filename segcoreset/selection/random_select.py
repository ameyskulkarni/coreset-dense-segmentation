"""The floor baseline (§5.1): uniform random subset, no features used."""
from __future__ import annotations

import random as _random
from typing import Sequence


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    rng = _random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    return sorted(ids[:budget])
