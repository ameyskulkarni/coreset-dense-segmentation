"""Patch balance (Idea 1): atypicality-weighted coverage of the patch vocabulary with
diminishing returns, maximized greedily. Label-free; DINOv2 patch prototypes stand in for
classes.

Every patch p of image i has a nearest vocabulary prototype l_p and a cosine distance d_p to
it (`patch_label` / `patch_dist` derivations, same vocabulary). Image i puts mass

    m_ij = sum over its patches p with l_p = j of (d_p / mean d)^gamma

on prototype j, and a subset S is scored by

    f(S) = sum_j ( sum_{i in S} m_ij )^alpha,      0 < alpha <= 1.

    alpha < 1  diminishing returns per prototype: the 15th near-duplicate image of the same
               unusual object adds much less than the first image of a new one (diversity),
               and over-represented prototypes saturate, which flattens the subset's pseudo-class
               pixel distribution (the mechanism the results point to). alpha = 1 is plain
               mass ranking with no diversity.
    gamma      atypicality tilt: an unusual patch (large d_p) counts for more than a typical one.
               gamma = 0 is pure pseudo-pixel balance; larger gamma moves toward patch-tail ranking.
               Typical prototypes keep nonzero weight, so ordinary street scenes are still covered.

f is monotone submodular, so greedy selection is the standard (1 - 1/e)-approximate maximizer.
Greedy is deterministic: the selection `seed` does not change the subset (ties break by row
order), so different seeds differ only in training.

Config: `configs/selection/patch-balance.yaml`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from ..features.embeddings import EmbeddingSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchBalanceConfig:
    alpha: float = 0.5  # concavity (diminishing returns); 1 = none
    gamma: float = 1.0  # atypicality tilt on each patch's mass; 0 = none

    @classmethod
    def from_cfg(cls, cfg) -> "PatchBalanceConfig":
        """Pick and validate the algorithm's fields from a `selection:` config.

        Raises:
            ValueError: If `alpha` is outside (0, 1] or `gamma` is negative.
        """
        out = cls(**{f.name: float(cfg[f.name]) for f in fields(cls) if f.name in cfg})
        if not 0.0 < out.alpha <= 1.0:
            raise ValueError(f"selection.alpha must be in (0, 1], got {out.alpha}")
        if out.gamma < 0.0:
            raise ValueError(f"selection.gamma must be >= 0, got {out.gamma}")
        return out


def prototype_mass(labels: np.ndarray, dist: np.ndarray, gamma: float, K: int | None = None) -> np.ndarray:
    """Per-image mass on every prototype: `M[i, j] = sum_{p in i, l_p = j} (d_p / mean d)^gamma`.

    Args:
        labels: `[N, P]` prototype id per patch (any numeric dtype holding integers).
        dist: `[N, P]` cosine distance per patch to that prototype.
        gamma: Atypicality exponent.
        K: Number of prototypes (default: max id + 1).

    Returns:
        `[N, K]` float64 masses.

    Raises:
        ValueError: On shape mismatches or non-integer labels.
    """
    L = np.asarray(labels)
    D = np.asarray(dist, dtype=np.float64)
    if L.shape != D.shape or L.ndim != 2:
        raise ValueError(f"labels {L.shape} and distances {D.shape} must be the same [N, P] shape")
    Li = np.rint(L).astype(np.int64)
    if not np.allclose(L, Li):
        raise ValueError("patch labels must be integer prototype ids")
    K = int(Li.max()) + 1 if K is None else K
    w = (D / D.mean()) ** gamma if gamma else np.ones_like(D)
    n = len(Li)
    flat = (np.arange(n)[:, None] * K + Li).ravel()
    return np.bincount(flat, weights=w.ravel(), minlength=n * K).reshape(n, K)


def greedy_concave_coverage(M: np.ndarray, budget: int, alpha: float) -> np.ndarray:
    """Greedily pick `budget` rows maximizing `sum_j (sum_{i in S} M[i, j])^alpha`.

    Args:
        M: `[N, K]` nonnegative masses.
        budget: Rows to pick, in [0, N].
        alpha: Exponent in (0, 1].

    Returns:
        `[budget]` row indices in pick order.
    """
    n = len(M)
    if not 0 <= budget <= n:
        raise ValueError(f"budget={budget} must be in [0, N={n}]")
    covered = np.zeros(M.shape[1])
    taken = np.zeros(n, dtype=bool)
    order = []
    for _ in range(budget):
        base = covered ** alpha
        gain = ((covered + M) ** alpha - base).sum(axis=1)
        gain[taken] = -np.inf
        i = int(np.argmax(gain))
        order.append(i)
        taken[i] = True
        covered += M[i]
    return np.asarray(order, dtype=np.int64)


def run(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> dict:
    """Build prototype masses, select greedily, return the selection plus diagnostics.

    Args:
        image_ids: Candidate ids.
        features: Dict from `load_selection_features`: `embedding` = `[N, P]` patch labels,
            `patch_scores` = `[N, P]` patch distances (same vocabulary).
        budget: Number of images to keep.
        seed: Unused (greedy is deterministic); kept for the common selector signature.
        cfg: A `PatchBalanceConfig` or a `selection:` config.

    Returns:
        `{"selected_ids", "order", "summary"}`.

    Raises:
        TypeError: If either matrix is missing.
        ValueError: On row-count mismatches or invalid config.
    """
    ids = list(image_ids)
    pcfg = cfg if isinstance(cfg, PatchBalanceConfig) else PatchBalanceConfig.from_cfg(cfg)
    if not isinstance(features, dict) or not isinstance(features.get("embedding"), np.ndarray) \
            or not isinstance(features.get("patch_scores"), np.ndarray):
        raise TypeError("patch_balance needs `embedding:` (patch_label) and `patch_scores:` (patch_dist) blocks "
                        "(see configs/selection/patch-balance.yaml)")
    L, D = features["embedding"], features["patch_scores"]
    if len(L) != len(ids):
        raise ValueError(f"patch labels have {len(L)} rows but there are {len(ids)} image ids")
    K = None
    if not isinstance(cfg, PatchBalanceConfig) and "embedding" in cfg:
        K = EmbeddingSpec.from_cfg(cfg["embedding"]).params.get("K")
    M = prototype_mass(L, D, pcfg.gamma, K)
    order = greedy_concave_coverage(M, budget, pcfg.alpha)

    pool = M.sum(axis=0)
    sub = M[order].sum(axis=0)
    ent = lambda v: float(-(p := v[v > 0] / v.sum()) @ np.log(p))  # noqa: E731
    summary = {
        "alpha": pcfg.alpha, "gamma": pcfg.gamma, "K": M.shape[1], "n_selected": len(order),
        "prototype_entropy_pool/subset": [round(ent(pool), 4), round(ent(sub), 4)] if len(order) else None,
        "prototypes_covered_pool/subset": [int((pool > 0).sum()), int((sub > 0).sum())],
    }
    log.info("patch_balance summary: %s", summary)
    return {"selected_ids": sorted(ids[i] for i in order), "order": order, "summary": summary}


def auto_name(cfg) -> str:
    """Name from every subset-changing setting, e.g. `patchbal-a50-g100-dinov2vitb14-patchlabel-K256-nsample200-seed0`.

    Raises:
        ValueError: If a feature block is missing or the two blocks use different vocabularies.
    """
    if "embedding" not in cfg or "patch_scores" not in cfg:
        raise ValueError("patch_balance needs `embedding:` (patch_label) and `patch_scores:` (patch_dist) blocks")
    emb, dist = EmbeddingSpec.from_cfg(cfg["embedding"]), EmbeddingSpec.from_cfg(cfg["patch_scores"])
    if (emb.backbone, emb.params, emb.split) != (dist.backbone, dist.params, dist.split):
        raise ValueError("embedding (patch_label) and patch_scores (patch_dist) must use the same backbone, "
                         f"vocabulary params and split; got {emb.describe()} vs {dist.describe()}")
    p = PatchBalanceConfig.from_cfg(cfg)
    return f"patchbal-a{p.alpha * 100:g}-g{p.gamma * 100:g}-{emb.tag()}"


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    """Registry entry point: the sorted ids of the `budget` images picked by patch balance."""
    return run(image_ids, features, budget, seed, cfg)["selected_ids"]
