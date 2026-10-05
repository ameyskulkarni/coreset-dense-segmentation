"""ADE20K (ADEChallengeData2016 layout), scale/diversity dataset (§2). Expects
`<root>/images/{training,validation}/*.jpg` and `<root>/annotations/{training,validation}/*.png`
with raw label values 0 (background, ignored) and 1..150 (classes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SegmentationDataset, Sample


def _load_class_names(root: Path) -> list[str]:
    """Read the 150 ADE20K class names from `<root>/objectInfo150.txt`.

    The file is tab-separated with a header row; the last column holds comma-separated
    synonyms, of which the first is used.

    Args:
        root: ADEChallengeData2016 root directory.

    Returns:
        150 class names in class-id order, or placeholder names (`class_0` ...) if the file
        is missing or yields no parsable rows.
    """
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
        """Load class names, then defer to `SegmentationDataset.__init__`.

        Args:
            cfg: The resolved `dataset:` sub-config (needs `root`).
            split: `"training"` or `"validation"`.
            subset_ids: Optional image ids to restrict to (see base class).
            transform: Optional joint transform (see base class).
        """
        self.class_names = _load_class_names(Path(cfg.root))
        super().__init__(cfg, split, subset_ids, transform)

    def _list_samples(self, split: str) -> list[Sample]:
        """List `images/<split>/*.jpg` paired with `annotations/<split>/<stem>.png`.

        Args:
            split: `"training"` or `"validation"`.

        Returns:
            One `Sample` per JPEG, sorted by filename; `image_id` is the filename stem.

        Raises:
            FileNotFoundError: If `images/<split>` does not exist.
        """
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
        """Shift raw labels down by one: 1..150 -> 0..149, and 0 (background) -> 255 (ignore).

        Args:
            raw: Raw annotation PNG values, `[H, W]`.

        Returns:
            `[H, W]` uint8 train ids.
        """
        raw = raw.astype(np.int32)
        label = raw - 1
        label[raw == 0] = 255
        return label.astype(np.uint8)
