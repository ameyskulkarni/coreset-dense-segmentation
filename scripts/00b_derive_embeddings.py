#!/usr/bin/env python
"""Derive per-image embeddings from the raw feature cache (`00_extract_features.py`) and
append them to `results/features/<dataset>/derived.parquet` with provenance columns.

Methods:
    cls          raw L2-normalized CLS vector (as cached)
    patch_mean   mean of the patch-token grid per image (then L2-normalized)
    kmeans_hist  per-image normalized histogram over K patch pseudo-classes (§5.3)
    patch_dist   per-image vector of every patch's cosine distance to its nearest pseudo-class
                 prototype (Proposal 1, "dense prototypicality"; row-major over the patch grid)

kmeans_hist and patch_dist share one patch vocabulary per (K, n_sample, seed): spherical
k-means on n_sample random patches per TRAIN image, saved to
results/features/<dataset>/vocab/ and reused (segcoreset/features/patch_vocab.py).

Usage:
    python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 --method patch_mean
    python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 --method kmeans_hist --k 256 --n-sample 200 --seed 0
    python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 --method patch_dist --k 256 --n-sample 200 --seed 0
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
from tqdm import tqdm

from segcoreset.features.derived import DerivedStore
from segcoreset.features.patch_vocab import VocabSpec, assign_patches, load_or_fit_vocabulary
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


def _vocab_assignments(store, ids, vocab_ctx, desc):
    """Load (or fit on train) the patch vocabulary, then assign every patch of every image.

    Args:
        store: Raw `FeatureStore` for the split being derived.
        ids: Image ids, in output row order.
        vocab_ctx: `(VocabSpec, dataset, train FeatureStore, train ids)`.
        desc: Progress-bar label.

    Yields:
        `(labels [P], distance [P])` per image, in `ids` order (see `patch_vocab.assign_patches`).
    """
    spec, dataset, train_store, train_ids = vocab_ctx
    vocab = load_or_fit_vocabulary(spec, dataset, train_store, train_ids)
    C = torch.from_numpy(vocab.centroids).to("cuda" if torch.cuda.is_available() else "cpu")
    for i in tqdm(ids, desc=desc):
        yield assign_patches(store.load(i), C)


def _kmeans_hist(store, ids, vocab_ctx, **_):
    """Derivation `kmeans_hist`: per-image histogram over K patch pseudo-classes (§5.3).

    The pseudo-classes are the shared patch vocabulary (`segcoreset/features/patch_vocab.py`):
    spherical k-means on `n_sample` random patches per train image (the audit #4 fix; the old
    streaming `MiniBatchKMeans` fit is gone). Every patch of every image is then assigned to
    its nearest prototype and counted.

    Args:
        store: Raw `FeatureStore` for the split.
        ids: Image ids, in output row order.
        vocab_ctx: `(VocabSpec, dataset, train FeatureStore, train ids)`.

    Returns:
        `(histograms [N, K] summing to 1 per row, source "patch", params {K, n_sample, seed})`.
    """
    spec = vocab_ctx[0]
    hists = []
    for labels, _ in _vocab_assignments(store, ids, vocab_ctx, f"histograms K={spec.K}"):
        h = np.bincount(labels, minlength=spec.K).astype(np.float32)
        hists.append(h / h.sum())
    return np.stack(hists), "patch", spec.params()


def _patch_dist(store, ids, vocab_ctx, **_):
    """Derivation `patch_dist`: every patch's cosine distance to its nearest vocabulary prototype.

    One row per image of length P = Hp*Wp (1,369 at 518x518), row-major over the patch grid
    (unsorted, so it can be drawn as a heatmap). Proposal 1 ("dense prototypicality") scores
    an image by the mean of the top-q fraction of this row
    (`selection/prototypicality/scoring.py::tail_mean`).

    Args:
        store: Raw `FeatureStore` for the split.
        ids: Image ids, in output row order.
        vocab_ctx: `(VocabSpec, dataset, train FeatureStore, train ids)`.

    Returns:
        `(distances [N, P], source "patch", params {K, n_sample, seed})`.
    """
    spec = vocab_ctx[0]
    rows = [d for _, d in _vocab_assignments(store, ids, vocab_ctx, f"patch distances K={spec.K}")]
    return np.stack(rows), "patch", spec.params()


METHODS = {"cls": _cls, "patch_mean": _patch_mean, "kmeans_hist": _kmeans_hist, "patch_dist": _patch_dist}


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
    parser.add_argument("--k", type=int, default=256, help="kmeans_hist/patch_dist: vocabulary size K")
    parser.add_argument("--n-sample", type=int, default=200,
                        help="kmeans_hist/patch_dist: patches sampled per train image to fit the vocabulary")
    parser.add_argument("--seed", type=int, default=0, help="kmeans_hist/patch_dist: vocabulary sample + k-means seed")
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
    train_dir = Path("results/features") / cfg.dataset.name / cfg.features.name / "train"
    train_store = raw if args.split == "train" else FeatureStore(train_dir)  # vocabularies are always fitted on train
    vocab_ctx = (VocabSpec(cfg.features.name, K=args.k, n_sample=args.n_sample, seed=args.seed),
                 cfg.dataset.name, train_store, sorted(train_store.cached_ids()))
    emb, source, params = METHODS[args.method](raw, ids, vocab_ctx=vocab_ctx)
    store = DerivedStore(cfg.dataset.name)
    store.add(ids, emb, split=args.split, backbone=cfg.features.name, source=source,
              method=args.method, params=params, config_hash=config_hash(cfg),
              image_paths=[path_of[i] for i in ids])
    print(f"Wrote {len(ids)} x {emb.shape[1]} '{args.method}' embeddings -> {store.path}")


if __name__ == "__main__":
    main()
