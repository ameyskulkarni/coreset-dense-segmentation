"""Step 4 of prototypicality: turn per-image scores into a kept set of exactly `budget`
images.

    keep="hard"  keep the LEAST prototypical images (largest distance) — the paper's setting
                 (prune the easy, redundant ones).
    keep="easy"  keep the MOST prototypical images — what the paper's theory predicts is
                 better when data is scarce (§3, Fig. 1).

Cluster balancing (`balance = b` in [0, 1]) is the label-free analogue of the paper's
"50% class balancing" (App. H), with k-means clusters standing in for classes: every
cluster c first gets a guaranteed quota of floor(b * n_c * budget / N) images — b times
the share it would keep under uniform pruning — filled by the same hard/easy ranking
inside the cluster. All remaining slots go to the best-ranked images globally.
b=0 is plain global ranking; b=1 prunes every cluster by the same fraction.

Pure numpy, deterministic: ties are broken by input row order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEEP_DIRECTIONS = ("hard", "easy")


@dataclass(frozen=True)
class PruneResult:
    selected: np.ndarray  # [budget] sorted row indices into the input
    order: np.ndarray  # [N] global keep-priority order (first = kept first)
    quota: np.ndarray  # [k] guaranteed slots per cluster (all zero when balance=0)

    @property
    def n_from_quota(self) -> int:
        """Number of kept images that came from per-cluster quotas (0 when `balance == 0`)."""
        return int(self.quota.sum())


def keep_priority(distance: np.ndarray, keep: str) -> np.ndarray:
    """Row indices sorted from 'keep first' to 'prune first' for the given direction.

    `keep="hard"` orders by descending distance, `keep="easy"` by ascending distance; ties
    are broken by row index, so the order is fully deterministic.

    Args:
        distance: `[N]` per-image distance to the nearest prototype.
        keep: `"hard"` or `"easy"`.

    Returns:
        `[N]` int array of row indices.

    Raises:
        ValueError: If `keep` is not in `KEEP_DIRECTIONS`.
    """
    if keep not in KEEP_DIRECTIONS:
        raise ValueError(f"keep must be one of {KEEP_DIRECTIONS}, got '{keep}'")
    primary = -distance if keep == "hard" else distance
    return np.lexsort((np.arange(len(distance)), primary))  # last key is primary


def cluster_quotas(cluster: np.ndarray, k: int, budget: int, balance: float) -> np.ndarray:
    """Guaranteed kept slots per cluster: `min(n_c, floor(balance * n_c * budget / N))`.

    Since each quota is at most `balance` times the cluster's proportional share, the
    quotas sum to at most `balance * budget <= budget`.

    Args:
        cluster: `[N]` cluster index per image, in [0, k).
        k: Number of clusters.
        budget: Total images to keep.
        balance: Balancing strength in [0, 1].

    Returns:
        `[k]` int array of per-cluster quotas.
    """
    n = len(cluster)
    sizes = np.bincount(cluster, minlength=k)
    return np.minimum(sizes, np.floor(balance * sizes * budget / n).astype(int))


def select_by_score(distance: np.ndarray, cluster: np.ndarray, k: int, budget: int,
                    keep: str = "hard", balance: float = 0.0) -> PruneResult:
    """Choose exactly `budget` images: per-cluster quotas first, then best-ranked globally.

    Each cluster's quota is filled with its own top-ranked members (by `keep_priority`);
    the remaining `budget - sum(quota)` slots go to the highest-priority images not yet
    chosen, across all clusters.

    Args:
        distance: `[N]` per-image distance to the nearest prototype.
        cluster: `[N]` nearest-prototype index per image.
        k: Number of clusters.
        budget: Images to keep, in [0, N].
        keep: `"hard"` (largest distance first) or `"easy"` (smallest first).
        balance: Cluster-balancing strength in [0, 1]; 0 = pure global ranking.

    Returns:
        A `PruneResult` with sorted `selected` indices, the global `order`, and `quota`.

    Raises:
        ValueError: On mismatched lengths, out-of-range `budget`/`balance`, or bad `keep`.
    """
    n = len(distance)
    if len(cluster) != n:
        raise ValueError(f"distance ({n}) and cluster ({len(cluster)}) must be the same length")
    if not 0 <= budget <= n:
        raise ValueError(f"budget={budget} must be in [0, N={n}]")
    if not 0.0 <= balance <= 1.0:
        raise ValueError(f"balance must be in [0, 1], got {balance}")

    order = keep_priority(distance, keep)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    chosen = np.zeros(n, dtype=bool)
    quota = cluster_quotas(cluster, k, budget, balance)
    for c in np.flatnonzero(quota):
        members = np.flatnonzero(cluster == c)
        chosen[members[np.argsort(rank[members])[: quota[c]]]] = True

    remaining = budget - int(chosen.sum())  # >= 0 since sum(quota) <= balance * budget
    rest = order[~chosen[order]]
    chosen[rest[:remaining]] = True

    selected = np.flatnonzero(chosen)
    assert len(selected) == budget, (len(selected), budget)
    return PruneResult(selected=selected, order=order, quota=quota)
