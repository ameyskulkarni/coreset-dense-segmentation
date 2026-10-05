#!/usr/bin/env python
"""Inspect / debug prototypicality selection without writing a subset (docs/prototypicality.md).

Runs exactly the same code path as `02_select.py --selection prototypicality-*` and dumps
what is normally hidden:

    <out>/<dataset>_<name>_s<seed>_scores.csv    one row per image: cluster, cosine distance,
                                                 keep-priority rank, selected (if --ratio)
    <out>/<dataset>_<name>_s<seed>_clusters.csv  one row per prototype: size, distance stats,
                                                 balancing quota, # selected
    stdout                                       summary + the most/least prototypical ids

With --k-sweep, also runs the label-free k-robustness proxy (plan §9: pick k without
training): rank stability across k-means seeds (Spearman of distances) and subset overlap
(Jaccard) across seeds and against the config's k -> <out>/<dataset>_<name>_k_sweep.csv.

Usage:
    python scripts/02b_inspect_prototypicality.py --dataset cityscapes --selection prototypicality-hard --ratio 0.2
    python scripts/02b_inspect_prototypicality.py --dataset cityscapes --selection prototypicality-hard --ratio 0.2 \\
        --k-sweep 5 10 20 50 100 --sweep-seeds 0 1 2
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from segcoreset.data.registry import build_dataset
from segcoreset.features.embeddings import DEFAULT_FEATURES_ROOT, EmbeddingSpec, load_embedding_matrix
from segcoreset.selection.prototypicality import PrototypicalityConfig, run
from segcoreset.selection.registry import selection_name
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging


def scores_frame(res, budget: int | None) -> pd.DataFrame:
    """Build the per-image scores table for a prototypicality run.

    Args:
        res: A `PrototypicalityResult`.
        budget: If not `None`, add a boolean `selected` column from `res.prune.selected`.

    Returns:
        One row per image (`image_id`, `cluster`, `distance`, `keep_priority_rank`,
        optional `selected`), sorted by keep-priority rank.
    """
    s = res.scores
    rank = np.empty(len(res.image_ids), dtype=int)
    rank[res.prune.order] = np.arange(len(rank))
    df = pd.DataFrame({
        "image_id": res.image_ids, "cluster": s.cluster,
        "distance": s.distance, "keep_priority_rank": rank,
    })
    if budget is not None:
        df["selected"] = False
        df.loc[res.prune.selected, "selected"] = True
    return df.sort_values("keep_priority_rank").reset_index(drop=True)


def clusters_frame(df: pd.DataFrame, quota: np.ndarray) -> pd.DataFrame:
    """Aggregate the per-image table into one row per prototype.

    Args:
        df: Output of `scores_frame`.
        quota: `[k]` per-cluster quotas from `res.prune.quota`.

    Returns:
        One row per cluster id 0..k-1 (empty clusters zero-filled) with `size`,
        `dist_mean`/`dist_min`/`dist_max`, optional `n_selected`, and `quota`.
    """
    agg = {"size": ("image_id", "size"), "dist_mean": ("distance", "mean"),
           "dist_min": ("distance", "min"), "dist_max": ("distance", "max")}
    if "selected" in df:
        agg["n_selected"] = ("selected", "sum")
    out = df.groupby("cluster").agg(**agg).reindex(range(len(quota)), fill_value=0)
    out["quota"] = quota
    return out.reset_index()


def k_sweep(ids, X, budget, base: PrototypicalityConfig, ks, seeds) -> pd.DataFrame:
    """Label-free robustness of the ranking/subset to k and to the k-means seed.

    Runs prototypicality for every (k, seed) pair, then per k reports cluster-size extremes,
    mean pairwise Spearman correlation of distances and Jaccard overlap of selected subsets
    across seeds, and the same two statistics against the reference k (the config's k if
    it is in `ks`, else the smallest swept k) at matching seeds.

    Args:
        ids: Candidate image ids.
        X: `[N, D]` embedding matrix.
        budget: Images to keep per run.
        base: Config whose fields other than `k` are held fixed.
        ks: k values to sweep (sorted).
        seeds: k-means seeds; cross-seed columns are NaN with fewer than two.

    Returns:
        One row per k.
    """
    runs = {(k, sd): run(ids, X, budget, sd, PrototypicalityConfig(**{**base.__dict__, "k": k}))
            for k in ks for sd in seeds}
    ref_k = base.k if base.k in ks else ks[0]

    def jaccard(a, b):
        """Jaccard similarity |a & b| / |a | b| of two id collections (0 if both empty)."""
        a, b = set(a), set(b)
        return len(a & b) / max(len(a | b), 1)

    rows = []
    for k in ks:
        pairs = list(itertools.combinations(seeds, 2))
        sizes = np.concatenate([runs[(k, sd)].scores.cluster_sizes() for sd in seeds])
        rows.append({
            "k": k,
            "min_cluster_size": int(sizes.min()),
            "median_cluster_size": float(np.median(sizes)),
            "spearman_across_seeds": np.mean([spearmanr(runs[(k, a)].scores.distance, runs[(k, b)].scores.distance)[0]
                                              for a, b in pairs]) if pairs else np.nan,
            "jaccard_across_seeds": np.mean([jaccard(runs[(k, a)].selected_ids, runs[(k, b)].selected_ids)
                                             for a, b in pairs]) if pairs else np.nan,
            f"spearman_vs_k{ref_k}": np.mean([spearmanr(runs[(k, sd)].scores.distance, runs[(ref_k, sd)].scores.distance)[0]
                                             for sd in seeds]),
            f"jaccard_vs_k{ref_k}": np.mean([jaccard(runs[(k, sd)].selected_ids, runs[(ref_k, sd)].selected_ids)
                                            for sd in seeds]),
        })
    return pd.DataFrame(rows)


def main():
    """CLI entry point: run prototypicality without writing a subset and dump diagnostics.

    Writes `<out-dir>/<dataset>_<name>_s<seed>_{scores,clusters}.csv`, prints a summary and
    the most/least prototypical ids, and with `--k-sweep` also writes `..._k_sweep.csv`
    (using a 20% budget if `--ratio` was not given).

    Raises:
        SystemExit: If the chosen selection config is not a prototypicality method.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection", default="prototypicality-hard")
    parser.add_argument("--ratio", type=float, default=None, help="also mark which images would be selected")
    parser.add_argument("--seed", type=int, default=0, help="k-means seed")
    parser.add_argument("--top", type=int, default=10, help="print this many most/least prototypical ids")
    parser.add_argument("--k-sweep", type=int, nargs="*", default=None, help="k values for the robustness proxy")
    parser.add_argument("--sweep-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--features-root", default=str(DEFAULT_FEATURES_ROOT))
    parser.add_argument("--out-dir", default="results/selection_debug")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(dataset=args.dataset, selection=args.selection, overrides=args.overrides)
    if cfg.selection.method != "prototypicality":
        raise SystemExit(f"--selection {args.selection} is method '{cfg.selection.method}', not prototypicality")

    ids = build_dataset(cfg.dataset, split=cfg.dataset.train_split).image_ids()
    X = load_embedding_matrix(cfg.dataset.name, ids, EmbeddingSpec.from_cfg(cfg.selection.embedding), root=args.features_root)
    budget = round(args.ratio * len(ids)) if args.ratio is not None else 0
    res = run(ids, X, budget, args.seed, cfg.selection)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = selection_name(cfg.selection)
    stem = f"{cfg.dataset.name}_{name}_s{args.seed}"
    df = scores_frame(res, budget if args.ratio is not None else None)
    df.to_csv(out / f"{stem}_scores.csv", index=False)
    cl = clusters_frame(df, res.prune.quota)
    cl.to_csv(out / f"{stem}_clusters.csv", index=False)

    by_dist = df.sort_values(["distance", "image_id"])
    print(f"\nSelection: {name}\nEmbeddings: {X.shape[0]} x {X.shape[1]}  |  config: {res.config}")
    print("Summary:", res.summary() if args.ratio is not None else {k: v for k, v in res.summary().items() if "selected" not in k})
    print(f"\nPer-cluster table ({out / f'{stem}_clusters.csv'}):\n{cl.to_string(index=False, float_format='%.4f')}")
    print(f"\n{args.top} MOST prototypical (easy):\n{by_dist.head(args.top)[['image_id', 'cluster', 'distance']].to_string(index=False)}")
    print(f"\n{args.top} LEAST prototypical (hard):\n{by_dist.tail(args.top)[['image_id', 'cluster', 'distance']].to_string(index=False)}")
    print(f"\nPer-image scores -> {out / f'{stem}_scores.csv'}")

    if args.k_sweep:
        sweep_budget = budget or round(0.2 * len(ids))
        sw = k_sweep(ids, X, sweep_budget, res.config, sorted(args.k_sweep), args.sweep_seeds)
        sw.to_csv(out / f"{cfg.dataset.name}_{name}_k_sweep.csv", index=False)
        print(f"\nk-robustness proxy (budget={sweep_budget}, seeds={args.sweep_seeds}):\n{sw.to_string(index=False, float_format='%.3f')}")


if __name__ == "__main__":
    main()
