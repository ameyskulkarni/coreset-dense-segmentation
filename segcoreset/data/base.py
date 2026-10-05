"""Base dataset contract shared by every dataset (§12.1). Subclasses implement only
`_list_samples` (path discovery) and `_encode_label` (raw PNG values -> contiguous train
ids); subset filtering, transform application, and the `Dataset` protocol are shared so
selection/training/eval never need to know which dataset they're looking at.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    label_path: Path


class SegmentationDataset(Dataset):
    num_classes: int
    ignore_index: int = 255

    def __init__(self, cfg, split: str, subset_ids: Sequence[str] | None = None, transform: Callable | None = None):
        """Discover the split's samples and optionally restrict them to a subset.

        Args:
            cfg: The resolved `dataset:` sub-config (needs at least `root`).
            split: Split name as it appears on disk (e.g. `"train"`, `"val"`, `"training"`).
            subset_ids: Image ids to keep (from a `results/subsets/*.json` file); duplicates
                are collapsed. Samples follow set-iteration order, which varies between
                processes (string hash seed), so same-seed runs on a subset are not
                batch-for-batch identical — accepted; seed-averaging absorbs it. `None` keeps
                every sample in the split, in sorted path order.
            transform: Joint `(PIL image, label ndarray) -> (image, label)` callable applied in
                `__getitem__` (see `data/transforms.py`). `None` returns raw PIL/ndarray pairs.

        Raises:
            ValueError: If any `subset_ids` entry is not an image id in this split.
        """
        self.cfg = cfg
        self.split = split
        self.transform = transform
        samples = self._list_samples(split)
        if subset_ids is not None:
            subset_ids = set(subset_ids)
            by_id = {s.image_id: s for s in samples}
            missing = subset_ids - by_id.keys()
            if missing:
                raise ValueError(
                    f"{len(missing)} subset ids not found in {self.__class__.__name__}/{split}, "
                    f"e.g. {sorted(missing)[:5]}"
                )
            samples = [by_id[i] for i in subset_ids]
        self.samples: list[Sample] = samples

    def _list_samples(self, split: str) -> list[Sample]:
        """Discover every (image, label) pair for `split` on disk. Implemented by subclasses.

        Args:
            split: Split name as it appears on disk.

        Returns:
            One `Sample` per image, in a deterministic (sorted) order.

        Raises:
            FileNotFoundError: (in subclasses) If the split directory does not exist.
        """
        raise NotImplementedError

    def _encode_label(self, raw: np.ndarray) -> np.ndarray:
        """Map raw label-PNG pixel values to contiguous ids in [0, num_classes), with
        `ignore_index` everywhere else. Implemented by subclasses.

        Args:
            raw: The label PNG as an integer `[H, W]` array.

        Returns:
            A `[H, W]` uint8 array of train ids / `ignore_index`.
        """
        raise NotImplementedError

    def image_ids(self) -> list[str]:
        """Return the image ids of this dataset's samples, in sample order.

        This is the candidate pool that selectors choose from and that subset files refer to.
        """
        return [s.image_id for s in self.samples]

    def __len__(self) -> int:
        """Return the number of samples (after any subset filtering)."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load, encode, and transform one sample.

        Args:
            idx: Index into `self.samples`.

        Returns:
            A dict with `"image"` (RGB PIL image, or a normalized `[3, H, W]` float tensor once
            `transform` includes `Normalize`), `"label"` (encoded `[H, W]` ndarray, or a long
            tensor after `Normalize`), and `"image_id"` (str).
        """
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        label = self._encode_label(np.array(Image.open(s.label_path)))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return {"image": image, "label": label, "image_id": s.image_id}
