"""Loss construction (§4)."""
from __future__ import annotations

import torch.nn as nn


def build_loss(dataset_cfg) -> nn.Module:
    """Plain per-pixel cross-entropy with the dataset's ignore index. No class weighting —
    keeping the loss identical across selection methods is what makes subset content the
    only variable; `class_balanced` (§5.2) reweights the DATA, not the loss."""
    return nn.CrossEntropyLoss(ignore_index=dataset_cfg.ignore_index)
