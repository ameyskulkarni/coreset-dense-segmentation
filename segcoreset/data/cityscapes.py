"""Cityscapes (fine annotations), main testbed (§2). Expects the official download layout:
`<root>/leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png` and
`<root>/gtFine/{train,val,test}/<city>/*_gtFine_labelIds.png`.

Maps the raw 34-class `labelIds` directly to the 19 Cityscapes eval classes (no need to
pre-run cityscapesScripts' `labelTrainIds` conversion) using the official mapping from
cityscapesScripts `labels.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import SegmentationDataset, Sample

_LABEL_ID_TO_TRAIN_ID: dict[int, int] = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255,
    17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
}

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def _build_lut() -> np.ndarray:
    """Build a 256-entry lookup table from raw Cityscapes `labelIds` to the 19 train ids.

    Every raw id not in `_LABEL_ID_TO_TRAIN_ID` (and every id mapped to 255 there) becomes
    255, so indexing the table with any uint8 label image is safe.

    Returns:
        A `[256]` uint8 array usable as `lut[raw_label]`.
    """
    lut = np.full(256, 255, dtype=np.uint8)
    for raw_id, train_id in _LABEL_ID_TO_TRAIN_ID.items():
        lut[raw_id] = train_id
    return lut


class CityscapesDataset(SegmentationDataset):
    num_classes = 19
    ignore_index = 255
    class_names = CLASS_NAMES
    _lut = _build_lut()

    def _list_samples(self, split: str) -> list[Sample]:
        """List `leftImg8bit/<split>/<city>/*_leftImg8bit.png` with their `gtFine_labelIds` labels.

        Args:
            split: `"train"`, `"val"`, or `"test"`.

        Returns:
            One `Sample` per image, sorted by path; `image_id` is the filename with the
            `_leftImg8bit.png` suffix removed (e.g. `aachen_000000_000019`).

        Raises:
            FileNotFoundError: If `leftImg8bit/<split>` does not exist.
        """
        root = Path(self.cfg.root)
        img_dir, lbl_dir = root / "leftImg8bit" / split, root / "gtFine" / split
        if not img_dir.exists():
            raise FileNotFoundError(f"Cityscapes split '{split}' not found at {img_dir}")
        samples = []
        for img_path in sorted(img_dir.glob("*/*_leftImg8bit.png")):
            city, stem = img_path.parent.name, img_path.name[: -len("_leftImg8bit.png")]
            lbl_path = lbl_dir / city / f"{stem}_gtFine_labelIds.png"
            samples.append(Sample(image_id=stem, image_path=img_path, label_path=lbl_path))
        return samples

    def _encode_label(self, raw: np.ndarray) -> np.ndarray:
        """Map raw 34-class `labelIds` to the 19 eval train ids via the precomputed LUT.

        Args:
            raw: Raw `*_gtFine_labelIds.png` values, `[H, W]`.

        Returns:
            `[H, W]` uint8 train ids, 255 for void/ignored classes.
        """
        return self._lut[raw]
