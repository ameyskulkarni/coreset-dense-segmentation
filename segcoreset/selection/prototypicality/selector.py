"""Prototypicality selector: glue between the common `select()` contract (`../base.py`)
and the two pure steps (`scoring.py` -> `pruning.py`).

`run()` returns everything (scores, clusters, priority order, summary) for debugging and
for `scripts/02b_inspect_prototypicality.py`; `select()` is the registry entry point and
returns only the chosen ids.

Two image scores (`score:`), with clustering and picking identical for both:

    centroid    cosine distance to the image's k-means prototype (Sorscher et al.; default)
    patch_tail  Proposal 1, "dense prototypicality": mean of the image's top-q fraction of
                patch distances to a patch vocabulary (`tail_mean`). The k clusters still
                come from `embedding:` and are used only for cluster balancing; the patch
                distances come from a second block, `patch_scores:`. So a patch_tail run
                differs from the centroid run on the same `embedding:`/k/seed in the score
                only (docs/proposals-explained.md, "Proposal 1").
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from ...features.embeddings import EmbeddingSpec
from .pruning import KEEP_DIRECTIONS, PruneResult, select_by_score
from .scoring import PrototypeScores, compute_prototype_scores, tail_mean

log = logging.getLogger(__name__)

SCORES = ("centroid", "patch_tail")


@dataclass(frozen=True)
class PrototypicalityConfig:
    k: int = 20
    keep: str = "hard"
    balance: float = 0.5  # default: each cluster keeps >= 50% of its fair share (see pruning.py)
    n_init: int = 10
    max_iter: int = 300
    score: str = "centroid"  # or "patch_tail" (Proposal 1; needs a `patch_scores:` block)
    q: float = 0.05  # patch_tail only: tail fraction of each image's patches that is averaged

    @classmethod
    def from_cfg(cls, cfg) -> "PrototypicalityConfig":
        """Pick the algorithm's fields out of a `selection:` config (ignores name/method/embedding).

        Fields absent from `cfg` keep their dataclass defaults.

        Args:
            cfg: The resolved `selection:` sub-config (or any mapping).

        Returns:
            The validated `PrototypicalityConfig`.

        Raises:
            ValueError: If `keep` is not in `KEEP_DIRECTIONS`, `balance` is outside [0, 1],
                `score` is not in `SCORES`, or `q` is outside (0, 1].
        """
        out = cls(**{f: cfg[f] for f in cls.__dataclass_fields__ if f in cfg})
        if out.keep not in KEEP_DIRECTIONS:
            raise ValueError(f"selection.keep must be one of {KEEP_DIRECTIONS}, got '{out.keep}'")
        if not 0.0 <= out.balance <= 1.0:
            raise ValueError(f"selection.balance must be in [0, 1], got {out.balance}")
        if out.score not in SCORES:
            raise ValueError(f"selection.score must be one of {SCORES}, got '{out.score}'")
        if not 0.0 < out.q <= 1.0:
            raise ValueError(f"selection.q must be in (0, 1], got {out.q}")
        return out


@dataclass(frozen=True)
class PrototypicalityResult:
    image_ids: list[str]  # input order
    selected_ids: list[str]  # sorted
    scores: PrototypeScores  # `distance` is the score actually ranked (the tail score for patch_tail)
    prune: PruneResult
    config: PrototypicalityConfig
    seed: int
    centroid_distance: np.ndarray | None = None  # patch_tail only: the replaced centroid distance

    def summary(self) -> dict:
        """Return a JSON-friendly summary of a selection for logging and debugging.

        Includes the config and seed, total/selected counts, cluster-size and distance
        min/median/max, the distance range of the selected images (`None` if nothing was
        selected), how many slots came from cluster quotas, how many clusters kept zero images,
        and the k-means inertia; for `patch_tail`, also the Spearman correlation between the
        tail score and the centroid distance it replaced.
        """
        d, sizes = self.scores.distance, self.scores.cluster_sizes()
        sel = self.prune.selected
        kept_per_cluster = np.bincount(self.scores.cluster[sel], minlength=self.scores.k)
        extra = {}
        if self.centroid_distance is not None:
            extra["spearman_score_vs_centroid_distance"] = round(float(spearmanr(d, self.centroid_distance)[0]), 4)
        return {
            **asdict(self.config), "seed": self.seed,
            "n_total": len(self.image_ids), "n_selected": len(sel),
            "cluster_size_min/median/max": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
            "distance_min/median/max": [round(float(x), 4) for x in (d.min(), np.median(d), d.max())],
            "selected_distance_min/max": [round(float(d[sel].min()), 4), round(float(d[sel].max()), 4)] if len(sel) else None,
            "n_from_cluster_quota": self.prune.n_from_quota,
            "clusters_with_zero_kept": int((kept_per_cluster == 0).sum()),
            "kmeans_inertia": round(self.scores.inertia, 4),
            **extra,
        }


def run(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> PrototypicalityResult:
    """Score and prune, returning every intermediate result (used by `select` and `02b`).

    Args:
        image_ids: Candidate ids, row-aligned with `features`.
        features: `[N, D]` embedding matrix, or a dict `{"embedding": [N, D], "patch_scores":
            [N, P]}` (`load_selection_features`); `patch_scores` is required for
            `score: patch_tail` and ignored otherwise.
        budget: Number of images to keep, in [0, N] (0 is allowed, e.g. for inspection only).
        seed: k-means seed.
        cfg: A `PrototypicalityConfig`, or a `selection:` config to build one from.

    Returns:
        A `PrototypicalityResult` (selected ids sorted; scores in input order).

    Raises:
        TypeError: If there is no embedding matrix (e.g. no `embedding:` block was configured,
            so `02_select.py` passed `None`), or `score: patch_tail` without patch scores.
        ValueError: If `features` and `image_ids` differ in length, or on invalid config.
    """
    ids = list(image_ids)
    pcfg = cfg if isinstance(cfg, PrototypicalityConfig) else PrototypicalityConfig.from_cfg(cfg)
    patch_dist = None
    if isinstance(features, dict):
        features, patch_dist = features.get("embedding"), features.get("patch_scores")
    if not isinstance(features, np.ndarray):
        raise TypeError("prototypicality needs an [N, D] embedding matrix as `features`; give the selection "
                        "config an `embedding:` block so 02_select.py loads one (see configs/selection/prototypicality-*.yaml)")
    if len(features) != len(ids):
        raise ValueError(f"features has {len(features)} rows but there are {len(ids)} image ids")
    if pcfg.score == "patch_tail":
        if not isinstance(patch_dist, np.ndarray):
            raise TypeError("score: patch_tail needs per-patch distances; give the selection config a `patch_scores:` "
                            "block (see configs/selection/prototypicality-hard-patchtail.yaml)")
        if len(patch_dist) != len(ids):
            raise ValueError(f"patch_scores has {len(patch_dist)} rows but there are {len(ids)} image ids")

    scores = compute_prototype_scores(features, k=pcfg.k, seed=seed, n_init=pcfg.n_init, max_iter=pcfg.max_iter)
    centroid_distance = None
    if pcfg.score == "patch_tail":  # same clusters (balancing), new score (ranking)
        centroid_distance = scores.distance
        scores = replace(scores, distance=tail_mean(patch_dist, pcfg.q))
    prune = select_by_score(scores.distance, scores.cluster, scores.k, budget, keep=pcfg.keep, balance=pcfg.balance)
    result = PrototypicalityResult(image_ids=ids, selected_ids=sorted(ids[i] for i in prune.selected),
                                   scores=scores, prune=prune, config=pcfg, seed=seed,
                                   centroid_distance=centroid_distance)
    log.info("prototypicality summary: %s", result.summary())
    return result


def auto_name(cfg) -> str:
    """Descriptive selection name for `name: auto` — every setting that changes the subset.

    Format: `proto-<keep>-k<k>-bal<balance x 100>-<embedding tag>`, plus `-ninit<n>` /
    `-maxiter<n>` only when those differ from the defaults, plus
    `-tailq<q x 100>-<patch_scores tag>` for `score: patch_tail` (centroid names are unchanged).
    Examples: `proto-hard-k20-bal50-dinov2vitb14-cls`,
    `proto-hard-k50-bal50-dinov2vitb14-patchmean-tailq5-dinov2vitb14-patchdist-K256-nsample200-seed0`.
    Contains no `_` (it is part of the subset filename).

    Args:
        cfg: The resolved `selection:` sub-config (must have an `embedding:` block).

    Returns:
        The name.

    Raises:
        ValueError: If `cfg` has no `embedding:` block (or no `patch_scores:` block for
            `score: patch_tail`), or on invalid prototypicality fields.
    """
    if "embedding" not in cfg:
        raise ValueError("prototypicality needs an `embedding:` block to build its name")
    p = PrototypicalityConfig.from_cfg(cfg)
    default = PrototypicalityConfig()
    name = f"proto-{p.keep}-k{p.k}-bal{p.balance * 100:g}-{EmbeddingSpec.from_cfg(cfg['embedding']).tag()}"
    if p.n_init != default.n_init:
        name += f"-ninit{p.n_init}"
    if p.max_iter != default.max_iter:
        name += f"-maxiter{p.max_iter}"
    if p.score == "patch_tail":
        if "patch_scores" not in cfg:
            raise ValueError("score: patch_tail needs a `patch_scores:` block to build its name")
        name += f"-tailq{p.q * 100:g}-{EmbeddingSpec.from_cfg(cfg['patch_scores']).tag()}"
    return name


def select(image_ids: Sequence[str], features, budget: int, seed: int, cfg) -> list[str]:
    """Registry entry point: return the sorted ids of the `budget` images kept by prototypicality.

    Args:
        image_ids: Candidate ids.
        features: `[N, D]` embeddings row-aligned with `image_ids` (or the dict `run` accepts).
        budget: Number of images to keep.
        seed: k-means seed (the only stochastic step).
        cfg: The resolved `selection:` sub-config (`k`, `keep`, `balance`, `n_init`,
            `max_iter`, `score`, `q`).

    Returns:
        The selected ids, sorted.
    """
    return run(image_ids, features, budget, seed, cfg).selected_ids
