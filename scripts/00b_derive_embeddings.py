#!/usr/bin/env python
"""Derive per-image embeddings from the raw feature cache (`00_extract_features.py`) and
append them to `results/features/<dataset>/derived.parquet` with provenance columns.

Methods:
    cls          raw L2-normalized CLS vector (as cached)
    patch_mean   mean of the patch-token grid per image (then L2-normalized)
    kmeans_hist  MiniBatchKMeans over ALL train patches -> K pseudo-classes; each image's
                 embedding is its normalized pseudo-class histogram (§5.3). Fit on --split train.

Usage:
    python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 --method patch_mean
    python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 --method kmeans_hist --k 256 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from segcoreset.features.derived import DerivedStore
from segcoreset.features.store import FeatureStore
from segcoreset.utils.config import build_config, config_hash
from segcoreset.utils.logging_setup import setup_logging


def _cls(store, ids, **_):
    """Derivation `cls`: stack each image's cached (already L2-normalized) CLS vector.

    Args:
        store: Raw `FeatureStore` for the split.
        ids: Image ids, in output row order.

    Returns:
        `(embeddings [N, D], source "cls", params {})`.
    """
    return np.stack([store.load(i)["cls"].numpy() for i in tqdm(ids, desc="cls")]), "cls", {}


def _patch_mean(store, ids, **_):
    """Derivation `patch_mean`: mean-pool each image's patch grid, then L2-normalize.

    Args:
        store: Raw `FeatureStore` for the split.
        ids: Image ids, in output row order.

    Returns:
        `(embeddings [N, D], source "patch", params {})`.
    """
    out = []
    for i in tqdm(ids, desc="patch_mean"):
        p = store.load(i)["patch"].float()  # [Hp, Wp, D]
        out.append(F.normalize(p.mean(dim=(0, 1)), dim=-1).numpy())
    return np.stack(out), "patch", {}


def _kmeans_hist(store, ids, k, seed, **_):
    """Derivation `kmeans_hist`: per-image histogram over `k` patch pseudo-classes (§5.3).

    Fits `MiniBatchKMeans` by streaming every image's patch tokens (one image per
    `partial_fit` call, so memory stays bounded), then assigns each image's patches to their
    nearest centroid and returns the normalized count histogram.

    Args:
        store: Raw `FeatureStore` for the split.
        ids: Image ids, in output row order (all used for fitting).
        k: Number of pseudo-classes (histogram bins).
        seed: k-means `random_state`.

    Returns:
        `(histograms [N, k] summing to 1 per row, source "patch", params {"K": k, "seed": seed})`.
    """
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=8192, n_init=3)
    for i in tqdm(ids, desc=f"fit kmeans K={k}"):  # streaming: one image's patches at a time
        p = store.load(i)["patch"]
        km.partial_fit(p.reshape(-1, p.shape[-1]).numpy())
    hists = []
    for i in tqdm(ids, desc="histograms"):
        p = store.load(i)["patch"]
        labels = km.predict(p.reshape(-1, p.shape[-1]).numpy())
        h = np.bincount(labels, minlength=k).astype(np.float32)
        hists.append(h / h.sum())
    return np.stack(hists), "patch", {"K": k, "seed": seed}


METHODS = {"cls": _cls, "patch_mean": _patch_mean, "kmeans_hist": _kmeans_hist}


def main():
    """CLI entry point: compute one derivation over a cached split and add it to `derived.parquet`.

    Raises:
        SystemExit: If the raw cache for the split is empty.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", default="dinov2_vitb14", help="raw representation config name, e.g. dinov2_vitb14")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--k", type=int, default=256, help="kmeans_hist: number of pseudo-classes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(dataset=args.dataset, features=args.features)
    raw_dir = Path("results/features") / cfg.dataset.name / cfg.features.name / args.split
    raw = FeatureStore(raw_dir)
    ids = sorted(raw.cached_ids())
    if not ids:
        raise SystemExit(f"No raw features in {raw_dir}; run 00_extract_features.py first.")

    with open(raw_dir / "index.json") as f:
        path_of = json.load(f)["image_paths"]  # written by 00_extract_features.py
    emb, source, params = METHODS[args.method](raw, ids, k=args.k, seed=args.seed)
    store = DerivedStore(cfg.dataset.name)
    store.add(ids, emb, split=args.split, backbone=cfg.features.name, source=source,
              method=args.method, params=params, config_hash=config_hash(cfg),
              image_paths=[path_of[i] for i in ids])
    print(f"Wrote {len(ids)} x {emb.shape[1]} '{args.method}' embeddings -> {store.path}")


if __name__ == "__main__":
    main()
