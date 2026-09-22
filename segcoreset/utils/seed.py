"""Deterministic seeding (§12.5). Captured in every `runs.csv` row via `recipe.seed_train`."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch RNGs. `deterministic=True` sets cuDNN to deterministic
    (but not `torch.use_deterministic_algorithms`, which breaks ops like adaptive pooling
    backward and interpolate) — a middle ground that keeps runs comparable without
    crashing on unsupported ops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
