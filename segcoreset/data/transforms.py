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
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomScale:
    """Resize image (bilinear) and label (nearest) by the same factor, drawn uniformly
    from `scale_range`."""

    def __init__(self, scale_range: tuple[float, float] = (0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, image: Image.Image, label: np.ndarray):
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
        self.h, self.w = size
        self.ignore_index = ignore_index
        self.pad_value = pad_value

    def __call__(self, image: Image.Image, label: np.ndarray):
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
        self.p = p

    def __call__(self, image: Image.Image, label: np.ndarray):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.ascontiguousarray(label[:, ::-1])
        return image, label


class PhotometricJitter:
    """Brightness/contrast/saturation/hue jitter on the image only (standard torchvision
    ColorJitter — deliberately not reimplemented, to avoid a subtle from-scratch bug)."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self._jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image: Image.Image, label: np.ndarray):
        return self._jitter(image), label


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: Image.Image, label: np.ndarray):
        image_t = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        label_t = torch.as_tensor(np.ascontiguousarray(label), dtype=torch.long)
        return image_t, label_t


def build_train_transform(dataset_cfg) -> Compose:
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
    return Compose([Normalize(dataset_cfg.mean, dataset_cfg.std)])
