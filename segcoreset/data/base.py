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
        raise NotImplementedError

    def _encode_label(self, raw: np.ndarray) -> np.ndarray:
        """Map raw label-PNG pixel values to contiguous ids in [0, num_classes), with
        `ignore_index` everywhere else."""
        raise NotImplementedError

    def image_ids(self) -> list[str]:
        return [s.image_id for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        label = self._encode_label(np.array(Image.open(s.label_path)))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return {"image": image, "label": label, "image_id": s.image_id}
