#!/usr/bin/env python
"""Silhouette and elbow (inertia) curves for an embedding — a diagnostic of how much
cluster structure the embedding space has, not a k selector used by any pipeline step.

Uses exactly the clustering prototypicality uses (`compute_prototype_scores`: L2-normalize,
k-means with the selection config's n_init/max_iter, nearest prototype by cosine), so the
curves describe the clusters selection would actually see. The embedding comes from the
selection config's `embedding:` block (override with --set, e.g. a different 00b method).

    <out>/<dataset>_<tag>_cluster_diagnostics.csv   one row per k: silhouette mean/std over
                                                    seeds, inertia mean/std, inertia drop
    <out>/<dataset>_<tag>_cluster_diagnostics.png   silhouette + elbow plot (needs matplotlib)

Reading it: silhouette is in [-1, 1]; below ~0.25 means no substantial cluster structure,
so its argmax is not meaningful. The "elbow" printed is the max-distance-to-chord point of
the inertia curve — a heuristic; a smooth curve with no visible bend has no real elbow.

Usage:
    python scripts/02c_cluster_diagnostics.py --dataset cityscapes
    python scripts/02c_cluster_diagnostics.py --dataset cityscapes --ks 2 3 5 10 20 50 100 --seeds 0 1 2 \\
        --set selection.embedding.method=patch_mean
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from segcoreset.data.registry import build_dataset
from segcoreset.features.embeddings import DEFAULT_FEATURES_ROOT, EmbeddingSpec, load_embedding_matrix
from segcoreset.selection.prototypicality import PrototypicalityConfig, compute_prototype_scores
from segcoreset.selection.prototypicality.scoring import l2_normalize
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging

DEFAULT_KS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]


def elbow_k(ks: np.ndarray, inertia: np.ndarray) -> int:
    """Return the k whose (log k, normalized inertia) point is farthest from the end-to-end chord.

    Args:
        ks: Sorted k values.
        inertia: Mean inertia per k.

    Returns:
        The heuristic elbow k (an endpoint if the curve has no bend).
    """
    x = np.log(ks)
    x = (x - x[0]) / (x[-1] - x[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    # chord from (0, 1) to (1, 0) is x + y = 1; distance is proportional to 1 - x - y
    return int(ks[np.argmax(1 - x - y)])


def cluster_diagnostics(X: np.ndarray, ks, seeds, base: PrototypicalityConfig, sample_size: int | None) -> pd.DataFrame:
    """Silhouette (cosine, on prototype assignments) and inertia for every (k, seed).

    Args:
        X: `[N, D]` embedding matrix.
        ks: k values to evaluate.
        seeds: k-means seeds; mean/std are taken over them.
        base: Supplies `n_init` / `max_iter` (k is replaced per run).
        sample_size: Silhouette subsample size (same seed-0 sample for every run), or `None`
            for all N points.

    Returns:
        One row per k.
    """
    Xn = l2_normalize(np.asarray(X, dtype=np.float64))
    sample = None
    if sample_size is not None and sample_size < len(Xn):
        sample = np.random.default_rng(0).choice(len(Xn), size=sample_size, replace=False)

    rows = []
    for k in ks:
        sils, inertias = [], []
        for sd in seeds:
            s = compute_prototype_scores(X, k=k, seed=sd, n_init=base.n_init, max_iter=base.max_iter)
            labels = s.cluster
            if len(np.unique(labels)) < 2:  # silhouette undefined for a single non-empty cluster
                sils.append(np.nan)
            elif sample is None:
                sils.append(silhouette_score(Xn, labels, metric="cosine"))
            else:
                sils.append(silhouette_score(Xn[sample], labels[sample], metric="cosine"))
            inertias.append(s.inertia)
        rows.append({"k": k, "silhouette_mean": np.nanmean(sils), "silhouette_std": np.nanstd(sils),
                     "inertia_mean": np.mean(inertias), "inertia_std": np.std(inertias)})
    df = pd.DataFrame(rows)
    df["inertia_drop_pct"] = -100 * df["inertia_mean"].pct_change()
    return df


def plot(df: pd.DataFrame, path: Path, title: str, elbow: int) -> bool:
    """Write a two-panel silhouette / elbow figure; return False if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.errorbar(df["k"], df["silhouette_mean"], yerr=df["silhouette_std"], marker="o", capsize=3)
    a.axhline(0.25, ls="--", c="gray", lw=1, label="0.25: weak-structure threshold")
    a.set(xscale="log", xlabel="k", ylabel="silhouette (cosine)", title="Silhouette")
    a.legend(fontsize=8)
    b.errorbar(df["k"], df["inertia_mean"], yerr=df["inertia_std"], marker="o", capsize=3)
    b.axvline(elbow, ls="--", c="gray", lw=1, label=f"heuristic elbow k={elbow}")
    b.set(xscale="log", xlabel="k", ylabel="inertia (k-means objective)", title="Elbow")
    b.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    """CLI entry point: compute and save silhouette/elbow curves for one embedding.

    Raises:
        SystemExit: If the selection config has no `embedding:` block, or fewer than two
            k values are given.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection", default="prototypicality-hard",
                        help="selection config whose `embedding:` block (and n_init/max_iter) to use")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--silhouette-sample", type=int, default=10000,
                        help="subsample this many points for silhouette (O(N^2)); 0 = use all")
    parser.add_argument("--tag", default=None, help="output filename tag; defaults to the embedding tag, e.g. dinov2vitb14-cls")
    parser.add_argument("--features-root", default=str(DEFAULT_FEATURES_ROOT))
    parser.add_argument("--out-dir", default="results/selection_debug")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(dataset=args.dataset, selection=args.selection, overrides=args.overrides)
    if "embedding" not in cfg.selection:
        raise SystemExit(f"--selection {args.selection} has no `embedding:` block")
    ks = sorted(set(args.ks))
    if len(ks) < 2:
        raise SystemExit("need at least two --ks values")

    spec = EmbeddingSpec.from_cfg(cfg.selection.embedding)
    ids = build_dataset(cfg.dataset, split=cfg.dataset.train_split).image_ids()
    X = load_embedding_matrix(cfg.dataset.name, ids, spec, root=args.features_root)
    base = PrototypicalityConfig.from_cfg(cfg.selection)

    df = cluster_diagnostics(X, ks, args.seeds, base, args.silhouette_sample or None)
    elbow = elbow_k(df["k"].to_numpy(), df["inertia_mean"].to_numpy())
    best = df.loc[df["silhouette_mean"].idxmax()]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or spec.tag()
    stem = f"{cfg.dataset.name}_{tag}_cluster_diagnostics"
    df.to_csv(out / f"{stem}.csv", index=False)

    print(f"\nEmbeddings: {X.shape[0]} x {X.shape[1]} from {spec.describe()}  |  seeds={args.seeds}")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\nsilhouette argmax: k={int(best['k'])} ({best['silhouette_mean']:.4f})"
          + ("  <- below 0.25: weak structure, argmax not meaningful" if best["silhouette_mean"] < 0.25 else ""))
    print(f"heuristic elbow:   k={elbow}  (max distance to chord on log-k; check the plot for a real bend)")
    print(f"\nTable -> {out / f'{stem}.csv'}")
    if plot(df, out / f"{stem}.png", f"{cfg.dataset.name} — {spec.describe()}", elbow):
        print(f"Plot  -> {out / f'{stem}.png'}")
    else:
        print("matplotlib not installed; skipped the plot")


if __name__ == "__main__":
    main()
