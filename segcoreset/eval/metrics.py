"""Metric primitives (§7): confusion-matrix-based mIoU/pixel accuracy, rare-class mIoU,
and boundary F-score. Pure numpy — usable identically from training-time quick eval and
standalone evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


class ConfusionMatrix:
    """Accumulates a C x C confusion matrix across images; derives per-class IoU, mIoU,
    pixel accuracy. Ignores pixels equal to `ignore_index`."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        """Start an all-zero C x C confusion matrix.

        Args:
            num_classes: Number of valid classes C (matrix is C x C, rows = ground truth,
                columns = prediction).
            ignore_index: Ground-truth value whose pixels are excluded from every count.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one image (or batch) of predictions into the matrix.

        Pixels whose target equals `ignore_index` are skipped. Predictions must lie in
        [0, num_classes); out-of-range values would corrupt the bincount.

        Args:
            pred: Predicted class ids, any shape.
            target: Ground-truth class ids, same shape as `pred`.
        """
        pred, target = pred.ravel(), target.ravel()
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        idx = target.astype(np.int64) * self.num_classes + pred.astype(np.int64)
        counts = np.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        """Compute IoU = TP / (TP + FP + FN) for each class.

        Returns:
            A `[num_classes]` float array; NaN for classes that never appear in either the
            ground truth or the predictions (so they are excluded from `nanmean`).
        """
        tp = np.diag(self.mat)
        fp = self.mat.sum(0) - tp
        fn = self.mat.sum(1) - tp
        denom = tp + fp + fn
        iou = np.full(self.num_classes, np.nan)
        valid = denom > 0
        iou[valid] = tp[valid] / denom[valid]
        return iou

    def miou(self) -> float:
        """Return the mean IoU over classes with a defined IoU (NaN classes skipped)."""
        return float(np.nanmean(self.per_class_iou()))

    def pixel_accuracy(self) -> float:
        """Return the fraction of non-ignored pixels predicted correctly (0.0 if none counted)."""
        return float(np.diag(self.mat).sum() / max(1, self.mat.sum()))


def rare_class_miou(per_class_iou: np.ndarray, rare_class_ids: list[int]) -> float:
    """The thesis metric (§7): mean IoU over the frozen bottom-K rare classes.

    Args:
        per_class_iou: `[num_classes]` IoUs from `ConfusionMatrix.per_class_iou`.
        rare_class_ids: Class ids from `load_rare_classes`.

    Returns:
        The NaN-ignoring mean over those classes (NaN if all of them are NaN).
    """
    return float(np.nanmean(per_class_iou[rare_class_ids]))


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    """1-pixel-wide boundary of a binary mask via morphological erosion.

    Pixels on the image border count as boundary (erosion pads with 0).

    Args:
        mask: Boolean `[H, W]` mask.

    Returns:
        Boolean `[H, W]` mask of pixels in `mask` with at least one 8-neighbor outside it.
    """
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & ~eroded


def boundary_f_score(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int,
                      tolerance_px: int = 3) -> float:
    """Boundary F-score at `tolerance_px` (§7 "trimap-IoU at 3px"), averaged over classes
    present in `target` for this image.

    For each such class, precision is the fraction of predicted boundary pixels within
    `tolerance_px` (Euclidean) of a ground-truth boundary pixel, and recall the converse;
    the class score is their harmonic mean. A class with a scorable ground-truth boundary
    but no predicted boundary scores 0.

    Ignore-labelled pixels have no correct answer, so — consistent with mIoU — they are kept
    out of the score: boundary pixels (predicted or ground truth) lying on an ignore pixel or
    8-adjacent to one are dropped before matching. A class whose every ground-truth boundary
    pixel touches ignore is skipped for that image.

    Why the 8-adjacent pixels too, not just the ignore pixels themselves: the two filters
    fix different halves of the problem.
      - On-ignore removal fixes PRECISION: the model predicts some class everywhere,
        including inside void regions, and edges it draws there (e.g. car|road inside a
        Cityscapes "ground" region) have no ground-truth edge nearby.
      - Adjacency removal fixes RECALL: `_boundary_mask(target == c)` only ever returns
        class-c pixels, never ignore pixels, so where class c meets void the GT boundary is
        the last row of labelled c pixels. On-ignore removal can't touch it. The model
        typically just continues predicting c into the void (road into "parking"), draws
        no edge there, and would be charged a miss for an edge against unlabelled pixels.
      - The adjacency rule is applied to predictions as well, for symmetry: once GT edges
        along void borders are dropped, predicted edges there must not count as spurious.
    Measured 2026-10-04 on the full-data Cityscapes SegFormer-B0 checkpoint
    (`cityscapes_segformer_b0_full_r1.0_s0_*`), 100 fixed val images (seed-0 subsample,
    ~13% ignore pixels), sliding-window eval:
        no ignore handling (old)    0.434
        on-ignore pixels only       0.475
        on-ignore + 8-adjacent      0.509   <- this implementation
    Synthetic sanity case (road | void band | road, model predicts road everywhere, i.e.
    perfect on every labelled pixel; ideal 1.0): 0.759 / 0.800 / 1.000 respectively.
    Images with no ignore pixels score identically under all three.

    Cost / trade-off: a 1-px strip around every void region is never scored, so a real
    semantic edge that happens to run along a void border (e.g. car next to a "dynamic"
    object) gets neither credit nor penalty. Treated as an acceptable, method-independent
    blind spot. To switch to the stricter on-ignore-only variant, set
    `near_ignore = ignored` below. Values computed before this fix (pre 2026-10-04 rows of
    `runs.csv`) are not comparable; rescore old checkpoints with `scripts/04_eval.py`.
    When writing up: "boundary pixels on or adjacent to ignore regions are excluded".

    Args:
        pred: `[H, W]` predicted class-id map.
        target: `[H, W]` ground-truth class-id map.
        num_classes: Number of classes to iterate over.
        ignore_index: Ground-truth value marking pixels to exclude (see above).
        tolerance_px: Distance tolerance in pixels.

    Returns:
        The mean F-score over scored classes, or NaN if no class was scored.
    """
    ignored = target == ignore_index
    near_ignore = binary_dilation(ignored, structure=np.ones((3, 3))) if ignored.any() else ignored
    scores = []
    for c in range(num_classes):
        gt_mask = target == c
        if not gt_mask.any():
            continue
        pred_mask = pred == c
        gt_b = _boundary_mask(gt_mask) & ~near_ignore
        pred_b = _boundary_mask(pred_mask) & ~near_ignore
        if not gt_b.any():
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
