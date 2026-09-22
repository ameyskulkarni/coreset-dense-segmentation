"""Evaluation protocol (§7): mIoU, rare-class mIoU, per-class IoU, boundary F-score, pixel
accuracy, all in one pass. The SAME `Evaluator` is used mid-training (fast, subsampled,
whole-image) and standalone (full protocol, sliding-window where the dataset calls for it)
— training just passes cheaper overrides so periodic eval doesn't dominate wall-clock time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data.rare_classes import load_rare_classes
from ..data.registry import build_dataset
from ..data.transforms import build_val_transform
from .metrics import ConfusionMatrix, boundary_f_score, rare_class_miou
from .sliding_window import sliding_window_inference


class Evaluator:
    def __init__(self, dataset_cfg, device: torch.device | None = None):
        self.cfg = dataset_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rare_path = Path(dataset_cfg.rare_classes_cache)
        self.rare_class_ids = load_rare_classes(rare_path) if rare_path.exists() else []

    @torch.no_grad()
    def evaluate(self, model, split: str | None = None, subset_ids=None, max_images: int | None = None,
                 compute_boundary_f: bool = True, mode: str | None = None,
                 resize_short_side: int | None = None) -> dict:
        model.eval()
        split = split or self.cfg.val_split
        ds = build_dataset(self.cfg, split=split, subset_ids=subset_ids, transform=build_val_transform(self.cfg))

        if max_images is not None and max_images < len(ds):
            rng = np.random.default_rng(0)  # fixed seed: periodic eval always samples the same subset
            idx = rng.choice(len(ds), size=max_images, replace=False)
            ds.samples = [ds.samples[i] for i in sorted(idx)]

        mode = mode or self.cfg.eval.get("mode", "whole_image")
        resize_short_side = resize_short_side if resize_short_side is not None else self.cfg.eval.get("resize_short_side")

        cm = ConfusionMatrix(self.cfg.num_classes, self.cfg.ignore_index)
        boundary_scores = []

        for i in range(len(ds)):
            item = ds[i]
            image, label = item["image"], item["label"].numpy()

            if mode == "sliding_window":
                pred = sliding_window_inference(
                    model, image, tuple(self.cfg.crop_size), tuple(self.cfg.eval.stride),
                    self.cfg.num_classes, self.device,
                ).numpy()
            else:
                inp, orig_size = image, image.shape[-2:]
                if resize_short_side:
                    scale = resize_short_side / min(orig_size)
                    inp = F.interpolate(inp.unsqueeze(0), scale_factor=scale, mode="bilinear",
                                         align_corners=False, recompute_scale_factor=False).squeeze(0)
                logits = model(inp.unsqueeze(0).to(self.device))
                logits = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
                pred = logits.argmax(1).squeeze(0).cpu().numpy()

            cm.update(pred, label)
            if compute_boundary_f:
                boundary_scores.append(boundary_f_score(pred, label, self.cfg.num_classes, self.cfg.ignore_index))

        per_class_iou = cm.per_class_iou()
        metrics = {
            "miou": cm.miou(),
            "pixel_acc": cm.pixel_accuracy(),
            "per_class_iou": {i: (None if np.isnan(v) else float(v)) for i, v in enumerate(per_class_iou)},
        }
        if self.rare_class_ids:
            metrics["rare_class_miou"] = rare_class_miou(per_class_iou, self.rare_class_ids)
        if compute_boundary_f and boundary_scores:
            metrics["boundary_f"] = float(np.nanmean(boundary_scores))
        return metrics
