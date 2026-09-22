"""CamVid, pilot/debug dataset (§2, Stage 0). Expects the classic SegNet-Tutorial 11-class
layout: `<root>/{train,val,test}/*.png` + `<root>/{train,val,test}annot/*.png`, label
values 0-10 (class) with 11 (or anything >= num_classes) treated as void.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SegmentationDataset, Sample

CLASS_NAMES = [
    "sky", "building", "pole", "road", "pavement", "tree",
    "sign_symbol", "fence", "car", "pedestrian", "bicyclist",
]

_SPLIT_DIRS = {"train": ("train", "trainannot"), "val": ("val", "valannot"), "test": ("test", "testannot")}


class CamVidDataset(SegmentationDataset):
    num_classes = 11
    ignore_index = 255
    class_names = CLASS_NAMES

    def _list_samples(self, split: str) -> list[Sample]:
        root = Path(self.cfg.root)
        img_sub, lbl_sub = _SPLIT_DIRS[split]
        img_dir, lbl_dir = root / img_sub, root / lbl_sub
        if not img_dir.exists():
            raise FileNotFoundError(f"CamVid split '{split}' not found at {img_dir}")
        samples = []
        for img_path in sorted(img_dir.glob("*.png")):
            samples.append(Sample(image_id=img_path.stem, image_path=img_path, label_path=lbl_dir / img_path.name))
        return samples

    def _encode_label(self, raw: np.ndarray) -> np.ndarray:
        label = raw.copy()
        label[label >= self.num_classes] = 255
        return label.astype(np.uint8)
