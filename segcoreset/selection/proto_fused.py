"""Fused prototypicality: rank-fuse the CLS, patch-mean and patch-tail hardness scores into
one image score, then prune with the unchanged prototypicality pruning step.

Each view gives an image a hardness (higher = less prototypical):

    cls    cosine distance to the nearest of `k_cls` spherical k-means prototypes of the
           CLS embedding (the `prototypicality-hard` CLS score)
    pmean  cosine distance to the nearest of `k` prototypes of the `embedding:` block
           (patch_mean; the `prototypicality-hard` patch-mean score)
    tail   mean of the image's top-q patch distances to the patch vocabulary (`tail_mean`)

Every view is turned into a percentile rank in (0, 1] (average ranks for ties), so scores
with different scales fuse fairly. `fusion` combines them with a FIXED rule (no learned
weights):

    meanrank  average of the view percentiles (Borda-style)
    minrank   minimum of the view percentiles: an image ranks high only if EVERY view finds
              it atypical (consensus)

Cluster balancing uses the `embedding:` clusters (patch-mean, k), exactly as the patch-tail
runs do, so at equal k/balance/seed a fused run differs from patch-tail in the score only.

This module does not modify `prototypicality/`; it reuses its pure functions. Config:
`configs/selection/prototypicality-hard-fused.yaml`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from ..features.embeddings import EmbeddingSpec
from .prototypicality.pruning import KEEP_DIRECTIONS, select_by_score
from .prototypicality.scoring import compute_prototype_scores, tail_mean

log = logging.getLogger(__name__)

FUSIONS = ("meanrank", "minrank")
VIEWS = ("cls", "pmean", "tail")


@dataclass(frozen=True)
class FusedConfig:
    k: int = 50  # prototypes of the `embedding:` block (balancing clusters + pmean view)
    k_cls: int = 20  # prototypes of the CLS embedding (cls view only)
    keep: str = "hard"
    balance: float = 0.5
    n_init: int = 10
    max_iter: int = 300
    q: float = 0.05  # tail view's patch fraction
    fusion: str = "meanrank"
    views: tuple = VIEWS

    @classmethod
    def from_cfg(cls, cfg) -> "FusedConfig":
        """Pick the algorithm's fields out of a `selection:` config and validate them.

        Raises:
            ValueError: On an unknown `keep`/`fusion`/view, `balance` outside [0, 1], or `q`
                outside (0, 1].
        """
        kw = {f.name: cfg[f.name] for f in fields(cls) if f.name in cfg}
        if "views" in kw:
            kw["views"] = tuple(kw["views"])
        out = cls(**kw)
        if out.keep not in KEEP_DIRECTIONS:
            raise ValueError(f"selection.keep must be one of {KEEP_DIRECTIONS}, got '{out.keep}'")
        if out.fusion not in FUSIONS:
            raise ValueError(f"selection.fusion must be one of {FUSIONS}, got '{out.fusion}'")
        if not out.views or any(v not in VIEWS for v in out.views) or len(set(out.views)) != len(out.views):
            raise ValueError(f"selection.views must be distinct entries of {VIEWS}, got {list(out.views)}")
        if not 0.0 <= out.balance <= 1.0:
            raise ValueError(f"selection.balance must be in [0, 1], got {out.balance}")
        if not 0.0 < out.q <= 1.0:
            raise ValueError(f"selection.q must be in (0, 1], got {out.q}")
        return out


def percentile(score: np.ndarray) -> np.ndarray:
    """Rank `score` ascending into percentiles in (0, 1] (ties share their average rank)."""
    return rankdata(score, method="average") / len(score)


def fuse(view_scores: dict[str, np.ndarray], fusion: str) -> np.ndarray:
    """Combine per-view hardness scores into one `[N]` score (higher = harder).

    Args:
        view_scores: `{view: [N] hardness}`.
        fusion: `meanrank` or `minrank`.

    Returns:
        `[N]` fused score in (0, 1].
    """
    R = np.stack([percentile(s) for s in view_scores.values()])
    return R.mean(axis=0) if fusion == "meanrank" else R.min(axis=0)


def run(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> dict:
    """Score every view, fuse, prune. Returns the selection plus per-view scores for debugging.

    Args:
        image_ids: Candidate ids.
        features: Dict from `load_selection_features`: `embedding` [N, D] (clustered; pmean
            view), `patch_scores` [N, P] (tail view), `extra["cls"]` [N, D] (cls view).
        budget: Number of images to keep.
        seed: Seeds both k-means runs.
        cfg: A `FusedConfig` or a `selection:` config.

    Returns:
        `{"selected_ids", "fused", "views", "cluster", "summary"}`.

    Raises:
        TypeError: If a view's input matrix is missing.
        ValueError: On row-count mismatches or invalid config.
    """
    ids = list(image_ids)
    fcfg = cfg if isinstance(cfg, FusedConfig) else FusedConfig.from_cfg(cfg)
    if not isinstance(features, dict) or not isinstance(features.get("embedding"), np.ndarray):
        raise TypeError("proto_fused needs `embedding:` + `patch_scores:` + `extra_embeddings: {cls: ...}` blocks "
                        "(see configs/selection/prototypicality-hard-fused.yaml)")
    X, P = features["embedding"], features.get("patch_scores")
    Xcls = (features.get("extra") or {}).get("cls")
    if "tail" in fcfg.views and not isinstance(P, np.ndarray):
        raise TypeError("the tail view needs a `patch_scores:` block")
    if "cls" in fcfg.views and not isinstance(Xcls, np.ndarray):
        raise TypeError("the cls view needs `extra_embeddings: {cls: <embedding block>}`")
    for name, M in (("embedding", X), ("patch_scores", P), ("extra.cls", Xcls)):
        if isinstance(M, np.ndarray) and len(M) != len(ids):
            raise ValueError(f"{name} has {len(M)} rows but there are {len(ids)} image ids")

    base = compute_prototype_scores(X, k=fcfg.k, seed=seed, n_init=fcfg.n_init, max_iter=fcfg.max_iter)
    views: dict[str, np.ndarray] = {}
    for v in fcfg.views:
        if v == "pmean":
            views[v] = base.distance
        elif v == "tail":
            views[v] = tail_mean(P, fcfg.q)
        else:
            views[v] = compute_prototype_scores(Xcls, k=fcfg.k_cls, seed=seed, n_init=fcfg.n_init,
                                                max_iter=fcfg.max_iter).distance
    score = fuse(views, fcfg.fusion)
    prune = select_by_score(score, base.cluster, base.k, budget, keep=fcfg.keep, balance=fcfg.balance)

    names = list(views)
    summary = {
        "fusion": fcfg.fusion, "views": names, "k": fcfg.k, "k_cls": fcfg.k_cls, "q": fcfg.q,
        "balance": fcfg.balance, "seed": seed, "n_selected": len(prune.selected),
        "spearman_between_views": {f"{a}~{b}": round(float(spearmanr(views[a], views[b])[0]), 4)
                                   for i, a in enumerate(names) for b in names[i + 1:]},
    }
    log.info("proto_fused summary: %s", summary)
    return {"selected_ids": sorted(ids[i] for i in prune.selected), "fused": score, "views": views,
            "cluster": base.cluster, "summary": summary}


def auto_name(cfg) -> str:
    """Name from every subset-changing setting, e.g.
    `protofused-meanrank-hard-k50-bal50-dinov2vitb14-patchmean-cls20-dinov2vitb14-cls-pmean-tailq5-dinov2vitb14-patchdist-K256-nsample200-seed0`.
    Views appear in config order; contains no `_`.

    Raises:
        ValueError: If a block the views need is missing, or on invalid fields.
    """
    if "embedding" not in cfg:
        raise ValueError("proto_fused needs an `embedding:` block to build its name")
    f = FusedConfig.from_cfg(cfg)
    default = FusedConfig()
    name = f"protofused-{f.fusion}-{f.keep}-k{f.k}-bal{f.balance * 100:g}-{EmbeddingSpec.from_cfg(cfg['embedding']).tag()}"
    if f.n_init != default.n_init:
        name += f"-ninit{f.n_init}"
    if f.max_iter != default.max_iter:
        name += f"-maxiter{f.max_iter}"
    for v in f.views:
        if v == "cls":
            if "cls" not in cfg.get("extra_embeddings", {}):
                raise ValueError("the cls view needs `extra_embeddings: {cls: ...}` to build its name")
            name += f"-cls{f.k_cls}-{EmbeddingSpec.from_cfg(cfg['extra_embeddings']['cls']).tag()}"
        elif v == "pmean":
            name += "-pmean"
        else:
            if "patch_scores" not in cfg:
                raise ValueError("the tail view needs a `patch_scores:` block to build its name")
            name += f"-tailq{f.q * 100:g}-{EmbeddingSpec.from_cfg(cfg['patch_scores']).tag()}"
    return name


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    """Registry entry point: the sorted ids of the `budget` images kept by fused prototypicality."""
    return run(image_ids, features, budget, seed, cfg)["selected_ids"]
