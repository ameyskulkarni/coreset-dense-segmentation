"""ADE20K (ADEChallengeData2016 layout), scale/diversity dataset (§2). Expects
`<root>/images/{training,validation}/*.jpg` and `<root>/annotations/{training,validation}/*.png`
with raw label values 0 (background, ignored) and 1..150 (classes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SegmentationDataset, Sample


def _load_class_names(root: Path) -> list[str]:
    info_path = root / "objectInfo150.txt"
    if not info_path.exists():
        return [f"class_{i}" for i in range(150)]
    names = []
    with open(info_path) as f:
        next(f)  # header: Idx Ratio Train Val Stuff Name
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                names.append(parts[-1].split(",")[0].strip())
    return names or [f"class_{i}" for i in range(150)]


class ADE20KDataset(SegmentationDataset):
    num_classes = 150
    ignore_index = 255

    def __init__(self, cfg, split, subset_ids=None, transform=None):
        self.class_names = _load_class_names(Path(cfg.root))
        super().__init__(cfg, split, subset_ids, transform)

    def _list_samples(self, split: str) -> list[Sample]:
        root = Path(self.cfg.root)
        img_dir, lbl_dir = root / "images" / split, root / "annotations" / split
        if not img_dir.exists():
            raise FileNotFoundError(f"ADE20K split '{split}' not found at {img_dir}")
        samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.png"
            samples.append(Sample(image_id=img_path.stem, image_path=img_path, label_path=lbl_path))
        return samples

    def _encode_label(self, raw: np.ndarray) -> np.ndarray:
        raw = raw.astype(np.int32)
        label = raw - 1
        label[raw == 0] = 255
        return label.astype(np.uint8)
