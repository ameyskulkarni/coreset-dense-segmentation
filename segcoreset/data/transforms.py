"""Joint image+label transforms (§2 preprocessing, fixed and identical across all runs).

Train: random scale [0.5, 2.0] -> random crop to the dataset's crop size -> random
horizontal flip -> photometric jitter (image only) -> normalize.
Val: normalize only — resizing/tiling for evaluation is the evaluator's job (§7), since
Cityscapes needs sliding-window tiling while ADE20K/CamVid use whole-image resize.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


class Compose:
    def __init__(self, transforms: Sequence):
        """Chain joint image+label transforms.

        Args:
            transforms: Joint transforms applied in order; each maps `(image, label)` to
                `(image, label)`.
        """
        self.transforms = transforms

    def __call__(self, image, label):
        """Apply each transform in sequence to the (image, label) pair.

        Returns:
            The final `(image, label)` pair.
        """
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomScale:
    """Resize image (bilinear) and label (nearest) by the same factor, drawn uniformly
    from `scale_range`."""

    def __init__(self, scale_range: tuple[float, float] = (0.5, 2.0)):
        """Configure the random rescale range.

        Args:
            scale_range: `(low, high)` bounds of the uniform scale factor.
        """
        self.scale_range = scale_range

    def __call__(self, image: Image.Image, label: np.ndarray):
        """Rescale image and label by one random factor (each side at least 1 px).

        Args:
            image: RGB PIL image.
            label: `[H, W]` encoded label array.

        Returns:
            The resized `(PIL image, label ndarray)`.
        """
        scale = random.uniform(*self.scale_range)
        w, h = image.size
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        label = np.array(Image.fromarray(label).resize((new_w, new_h), Image.NEAREST))
        return image, label


class RandomCrop:
    """Random crop to `size` = (h, w); pads with the dataset mean pixel (image) /
    `ignore_index` (label) first if the input is smaller than the crop."""

    def __init__(self, size: tuple[int, int], ignore_index: int, pad_value: tuple[int, int, int]):
        """Configure crop size and padding values.

        Args:
            size: Crop size `(h, w)`.
            ignore_index: Label value used to pad labels smaller than the crop.
            pad_value: RGB fill (0-255) used to pad images smaller than the crop.
        """
        self.h, self.w = size
        self.ignore_index = ignore_index
        self.pad_value = pad_value

    def __call__(self, image: Image.Image, label: np.ndarray):
        """Pad bottom/right if needed, then take a uniformly random `size` crop of both inputs.

        Args:
            image: RGB PIL image.
            label: `[H, W]` encoded label array.

        Returns:
            `(PIL image, label ndarray)`, both exactly `size`.
        """
        w, h = image.size
        pad_h, pad_w = max(0, self.h - h), max(0, self.w - w)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=self.pad_value)
            label = np.pad(label, ((0, pad_h), (0, pad_w)), constant_values=self.ignore_index)
            w, h = image.size
        top = random.randint(0, h - self.h)
        left = random.randint(0, w - self.w)
        image = image.crop((left, top, left + self.w, top + self.h))
        label = label[top : top + self.h, left : left + self.w]
        return image, label


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        """Configure the flip probability.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, image: Image.Image, label: np.ndarray):
        """With probability `p`, mirror both image and label left-right.

        Args:
            image: RGB PIL image.
            label: `[H, W]` encoded label array.

        Returns:
            The (possibly flipped) `(PIL image, contiguous label ndarray)`.
        """
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.ascontiguousarray(label[:, ::-1])
        return image, label


class PhotometricJitter:
    """Brightness/contrast/saturation/hue jitter on the image only (standard torchvision
    ColorJitter — deliberately not reimplemented, to avoid a subtle from-scratch bug)."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        """Wrap a torchvision `ColorJitter` with the given ranges.

        Args:
            brightness: `ColorJitter` brightness factor range.
            contrast: `ColorJitter` contrast factor range.
            saturation: `ColorJitter` saturation factor range.
            hue: `ColorJitter` hue shift range (in [0, 0.5]).
        """
        self._jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image: Image.Image, label: np.ndarray):
        """Color-jitter the image; return the label unchanged.

        Args:
            image: RGB PIL image.
            label: `[H, W]` encoded label array.

        Returns:
            `(jittered PIL image, label)`.
        """
        return self._jitter(image), label


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        """Store the per-channel normalization statistics.

        Args:
            mean: Per-channel RGB mean in [0, 1] units.
            std: Per-channel RGB std in [0, 1] units.
        """
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: Image.Image, label: np.ndarray):
        """Convert to tensors: image to normalized float, label to long.

        Args:
            image: RGB PIL image.
            label: `[H, W]` encoded label array.

        Returns:
            `(image [3, H, W] float32, label [H, W] int64)` tensors.
        """
        image_t = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        label_t = torch.as_tensor(np.ascontiguousarray(label), dtype=torch.long)
        return image_t, label_t


def build_train_transform(dataset_cfg) -> Compose:
    """Build the fixed training augmentation pipeline for a dataset.

    Random scale [0.5, 2.0] -> random crop to `dataset_cfg.crop_size` (padding with the
    dataset's mean pixel / `ignore_index`) -> horizontal flip (p=0.5) -> photometric jitter
    (unless `dataset_cfg.photometric_jitter` is false) -> normalize.

    Args:
        dataset_cfg: The resolved `dataset:` sub-config (needs `crop_size`, `mean`, `std`,
            `ignore_index`; optional `photometric_jitter`, default `True`).

    Returns:
        A `Compose` producing `(image tensor, label tensor)`.
    """
    crop_h, crop_w = dataset_cfg.crop_size
    pad_value = tuple(int(round(255 * m)) for m in dataset_cfg.mean)
    ops = [
        RandomScale((0.5, 2.0)),
        RandomCrop((crop_h, crop_w), ignore_index=dataset_cfg.ignore_index, pad_value=pad_value),
        RandomHorizontalFlip(0.5),
    ]
    if dataset_cfg.get("photometric_jitter", True):
        ops.append(PhotometricJitter())
    ops.append(Normalize(dataset_cfg.mean, dataset_cfg.std))
    return Compose(ops)


def build_val_transform(dataset_cfg) -> Compose:
    """Build the evaluation transform: normalization only, at native resolution.

    Resizing (whole-image mode) or tiling (sliding-window mode) is the evaluator's job.

    Args:
        dataset_cfg: The resolved `dataset:` sub-config (needs `mean`, `std`).

    Returns:
        A `Compose` producing `(image tensor, label tensor)`.
    """
    return Compose([Normalize(dataset_cfg.mean, dataset_cfg.std)])
