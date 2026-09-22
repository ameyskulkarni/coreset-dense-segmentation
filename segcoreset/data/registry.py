"""Dataset registry — adding a dataset is one new file + one line here."""
from __future__ import annotations

from .ade20k import ADE20KDataset
from .camvid import CamVidDataset
from .cityscapes import CityscapesDataset

DATASET_REGISTRY = {
    "cityscapes": CityscapesDataset,
    "ade20k": ADE20KDataset,
    "camvid": CamVidDataset,
}


def build_dataset(dataset_cfg, split: str, subset_ids=None, transform=None):
    name = dataset_cfg.name
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](dataset_cfg, split, subset_ids=subset_ids, transform=transform)
