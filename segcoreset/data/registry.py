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
    """Instantiate the dataset class registered under `dataset_cfg.name`.

    Args:
        dataset_cfg: The resolved `dataset:` sub-config.
        split: On-disk split name (usually `dataset_cfg.train_split` / `val_split`).
        subset_ids: Optional image ids to restrict to.
        transform: Optional joint image+label transform.

    Returns:
        A `SegmentationDataset` subclass instance.

    Raises:
        ValueError: If `dataset_cfg.name` is not in `DATASET_REGISTRY`.
    """
    name = dataset_cfg.name
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](dataset_cfg, split, subset_ids=subset_ids, transform=transform)
