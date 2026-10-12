#!/usr/bin/env python
"""Label-free rare-concept coverage of subset files: an offline (no training) proxy used to
compare selections and choose knob values (docs/proposals-explained.md, "The label-free check").

Concepts are the pseudo-classes of a patch vocabulary, read from the `kmeans_hist` derivation
(normalized per-image histograms x patches per image = per-image concept counts). The rarest
`--rare-frac` of concepts by total train patch count are the *rare concepts*. Per subset:

    rare_covered       # rare concepts with >= --min-patches patches in the subset
    rare_patch_share   share of all rare-concept patches that the subset contains
    lift               rare_patch_share / (n_images / N)   (1.0 = what a random subset gets in expectation)

Use a vocabulary INDEPENDENT of the one any selector optimizes (default K=1024, seed 1;
Proposal 1/2 use K=256, seed 0), otherwise a coverage selector wins this check by
construction. Validate the check against trained rare-class mIoU of existing subsets
before trusting it to pick knobs.

    <out>/<dataset>_concept_coverage_<vocab tag>.csv   one row per subset file
    stdout                                            mean +- std over seeds per (name, ratio)

Create the vocabulary's histograms first:
    python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 \\
        --method kmeans_hist --k 1024 --n-sample 200 --seed 1

Usage:
    python scripts/02e_concept_coverage.py --dataset cityscapes
    python scripts/02e_concept_coverage.py --dataset cityscapes --ratios 0.1 0.2 0.3 \\
        --subsets 'results/subsets/cityscapes_random_*.json' 'results/subsets/cityscapes_proto-hard-k50-*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from segcoreset.data.registry import build_dataset
from segcoreset.features.embeddings import DEFAULT_FEATURES_ROOT, EmbeddingSpec, load_embedding_matrix
from segcoreset.features.store import FeatureStore
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging


def main():
    """CLI entry point: score every matching subset file and write/print the coverage table.

    Raises:
        SystemExit: If no subset file matches.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", default="dinov2_vitb14")
    parser.add_argument("--subsets", nargs="*", default=None, help="glob patterns (default: all of the dataset's subsets)")
    parser.add_argument("--ratios", type=float, nargs="*", default=None, help="keep only these ratios")
    parser.add_argument("--k", type=int, default=1024, help="vocabulary size of the kmeans_hist derivation")
    parser.add_argument("--n-sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1, help="vocabulary seed (keep it different from the selectors')")
    parser.add_argument("--rare-frac", type=float, default=0.25, help="fraction of concepts (rarest first) called rare")
    parser.add_argument("--min-patches", type=int, default=20, help="patches a rare concept needs to count as covered")
    parser.add_argument("--features-root", default=str(DEFAULT_FEATURES_ROOT))
    parser.add_argument("--out-dir", default="results/selection_debug")
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(dataset=args.dataset, features=args.features)
    ids = build_dataset(cfg.dataset, split=cfg.dataset.train_split).image_ids()
    row_of = {i: r for r, i in enumerate(ids)}
    spec = EmbeddingSpec(backbone=cfg.features.name, method="kmeans_hist",
                         params={"K": args.k, "n_sample": args.n_sample, "seed": args.seed})
    H = load_embedding_matrix(cfg.dataset.name, ids, spec, root=args.features_root)
    raw = FeatureStore(Path(args.features_root) / cfg.dataset.name / cfg.features.name / "train")
    grid = raw.load(ids[0])["patch"].shape
    counts = np.rint(H * grid[0] * grid[1])  # [N, K] patches of each concept per image

    totals = counts.sum(axis=0)
    order = np.argsort(totals, kind="stable")
    order = order[totals[order] > 0]
    rare = order[: max(1, int(round(args.rare_frac * len(order))))]
    rare_total = totals[rare].sum()

    patterns = args.subsets or [f"results/subsets/{cfg.dataset.name}_*.json"]
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    rows = []
    for f in files:
        with open(f) as fh:
            meta = json.load(fh)
        if args.ratios and not any(np.isclose(meta.get("ratio", -1), r) for r in args.ratios):
            continue
        idx = [row_of[i] for i in meta["image_ids"]]
        c = counts[idx].sum(axis=0)[rare]
        share = c.sum() / rare_total
        rows.append({"file": Path(f).name, "name": meta.get("name", meta.get("method")), "ratio": meta.get("ratio"),
                     "seed": meta.get("seed"), "n_images": len(idx),
                     "rare_covered": int((c >= args.min_patches).sum()), "n_rare": len(rare),
                     "rare_patch_share": share, "lift": share / (len(idx) / len(ids))})
    if not rows:
        raise SystemExit(f"No subset files matched {patterns} (ratios {args.ratios})")

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.dataset.name}_concept_coverage_{spec.tag()}.csv"
    df.to_csv(path, index=False)
    print(f"Vocabulary {spec.describe()}: {len(rare)} rare concepts of {len(order)}, "
          f"{int(rare_total)} rare patches; covered = >= {args.min_patches} patches in the subset")
    summary = df.groupby(["ratio", "name"])[["rare_covered", "rare_patch_share", "lift"]].agg(["mean", "std"])
    print(summary.to_string(float_format="%.3f"))
    print(f"\nPer-subset rows -> {path}")


if __name__ == "__main__":
    main()
