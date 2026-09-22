"""Metric primitives (§7): confusion-matrix-based mIoU/pixel accuracy, rare-class mIoU,
and boundary F-score. Pure numpy — usable identically from training-time quick eval and
standalone evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


class ConfusionMatrix:
    """Accumulates a C x C confusion matrix across images; derives per-class IoU, mIoU,
    pixel accuracy. Ignores pixels equal to `ignore_index`."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred, target = pred.ravel(), target.ravel()
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        idx = target.astype(np.int64) * self.num_classes + pred.astype(np.int64)
        counts = np.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.mat)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = tp + fp + fn
        iou = np.full(self.num_classes, np.nan)
        valid = denom > 0
        iou[valid] = tp[valid] / denom[valid]
        return iou

    def miou(self) -> float:
        return float(np.nanmean(self.per_class_iou()))

    def pixel_accuracy(self) -> float:
        return float(np.diag(self.mat).sum() / max(1, self.mat.sum()))


def rare_class_miou(per_class_iou: np.ndarray, rare_class_ids: list[int]) -> float:
    """The thesis metric (§7): mean IoU over the frozen bottom-K rare classes."""
    return float(np.nanmean(per_class_iou[rare_class_ids]))


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """1-pixel-wide boundary of a binary mask via morphological erosion."""
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & ~eroded


def boundary_f_score(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int,
                      tolerance_px: int = 3) -> float:
    """Boundary F-score at `tolerance_px` (§7 "trimap-IoU at 3px"), averaged over classes
    present in `target` for this image. `pred`/`target`: [H, W] class-id maps."""
    scores = []
    for c in range(num_classes):
        gt_mask = target == c
        if not gt_mask.any():
            continue
        pred_mask = pred == c
        gt_b, pred_b = _boundary_mask(gt_mask), _boundary_mask(pred_mask)
        if not gt_b.any() and not pred_b.any():
            continue
        if not pred_b.any():
            scores.append(0.0)
            continue
        gt_dist = distance_transform_edt(~gt_b)
        pred_dist = distance_transform_edt(~pred_b)
        precision = float((gt_dist[pred_b] <= tolerance_px).mean())
        recall = float((pred_dist[gt_b] <= tolerance_px).mean())
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else float("nan")
