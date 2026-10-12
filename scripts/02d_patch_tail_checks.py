#!/usr/bin/env python
"""Offline (no training) checks for Proposal 1, "dense prototypicality"
(configs/selection/prototypicality-hard-patchtail.yaml; docs/proposals-explained.md, "Proposal 1").

    <out>/<dataset>_p1_<vocab tag>_vs_centroid.csv
        one row per (q, ratio, seed): how different the patch-tail selection is from the
        centroid selection it replaces (same embedding, k, keep, balance, seed = Exp 9):
        Spearman(tail score, centroid distance) and Jaccard(subsets). Also whether the
        in-memory centroid subset equals the committed Exp 9 subset file (regression check).
    <out>/<dataset>_p1_<vocab tag>_vocab_sensitivity.csv
        one row per (alternative vocabulary, q): Spearman of the tail scores against the base
        vocabulary's, and the mean Jaccard of the selected subsets over ratios x seeds.
    <out>/<dataset>_p1_vocab_health.csv
        one row per fitted vocabulary: cluster sizes (min / median / max, # tiny) + objective.

Alternative vocabularies are given as overrides of the base `patch_scores.params`, e.g.
`K=64 K=1024 n_sample=400 seed=1`; each must already be derived (the script prints the
`00b` command for any that is missing and skips it).

Usage:
    python scripts/02d_patch_tail_checks.py --dataset cityscapes
    python scripts/02d_patch_tail_checks.py --dataset cityscapes --qs 1.0 0.2 0.05 0.01 \\
        --alt-vocabs K=64 K=1024 n_sample=400 seed=1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from segcoreset.data.registry import build_dataset
from segcoreset.features.embeddings import DEFAULT_FEATURES_ROOT, EmbeddingSpec, load_embedding_matrix, load_selection_features
from segcoreset.features.patch_vocab import VocabSpec, load_vocabulary
from segcoreset.selection.io import load_subset, subset_path
from segcoreset.selection.prototypicality import PrototypicalityConfig, compute_prototype_scores, select_by_score, tail_mean
from segcoreset.selection.registry import selection_name
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging


def jaccard(a, b) -> float:
    """Jaccard similarity |a & b| / |a | b| of two index collections (0 if both empty)."""
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def parse_override(s: str) -> tuple[str, int]:
    """`"K=64"` -> `("K", 64)`; only the vocabulary params K / n_sample / seed are allowed."""
    key, _, val = s.partition("=")
    if key not in ("K", "n_sample", "seed") or not val:
        raise ValueError(f"--alt-vocabs entries look like K=64, n_sample=400 or seed=1; got '{s}'")
    return key, int(val)


def main():
    """CLI entry point: write the three CSVs described in the module docstring and print them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection", default="prototypicality-hard-patchtail")
    parser.add_argument("--qs", type=float, nargs="+", default=[1.0, 0.2, 0.05, 0.01])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alt-vocabs", nargs="*", default=["K=64", "K=1024", "n_sample=400", "seed=1"])
    parser.add_argument("--features-root", default=str(DEFAULT_FEATURES_ROOT))
    parser.add_argument("--out-dir", default="results/selection_debug")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(dataset=args.dataset, selection=args.selection, overrides=args.overrides)
    sel = cfg.selection
    if sel.method != "prototypicality" or sel.get("score") != "patch_tail" or "patch_scores" not in sel:
        raise SystemExit(f"--selection {args.selection} must be prototypicality with score: patch_tail and a patch_scores: block")
    p = PrototypicalityConfig.from_cfg(sel)
    ds_name = cfg.dataset.name
    ids = build_dataset(cfg.dataset, split=cfg.dataset.train_split).image_ids()
    feats = load_selection_features(ds_name, ids, sel, root=args.features_root)
    X, D = feats["embedding"], feats["patch_scores"]
    base_spec = EmbeddingSpec.from_cfg(sel.patch_scores)
    tails = {q: tail_mean(D, q) for q in args.qs}
    budgets = {r: round(r * len(ids)) for r in args.ratios}

    # Centroid reference (= Exp 9 for the patch-mean k50 config): same clusters, centroid score.
    ref_cfg = OmegaConf.merge(sel, {"score": "centroid"})
    ref_name = selection_name(ref_cfg)
    clusters, picks, rows = {}, {}, []
    for seed in args.seeds:
        s = compute_prototype_scores(X, k=p.k, seed=seed, n_init=p.n_init, max_iter=p.max_iter)
        clusters[seed] = s
        for r, b in budgets.items():
            ref = select_by_score(s.distance, s.cluster, s.k, b, keep=p.keep, balance=p.balance).selected
            committed = subset_path(ref_name, ds_name, r, seed)
            same = sorted(ids[i] for i in ref) == load_subset(committed) if committed.exists() else None
            for q, t in tails.items():
                sel_q = select_by_score(t, s.cluster, s.k, b, keep=p.keep, balance=p.balance).selected
                picks[(base_spec.tag(), q, r, seed)] = sel_q
                rows.append({"q": q, "ratio": r, "seed": seed,
                             "spearman_tail_vs_centroid": spearmanr(t, s.distance)[0],
                             "jaccard_vs_centroid_subset": jaccard(sel_q, ref),
                             "centroid_subset_matches_committed": same})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{ds_name}_p1_{base_spec.tag()}"
    vs = pd.DataFrame(rows)
    vs.to_csv(out / f"{stem}_vs_centroid.csv", index=False)
    print(f"\nReference (centroid) selection: {ref_name}")
    print(f"Patch-tail vs centroid ({out / f'{stem}_vs_centroid.csv'}):")
    print(vs.groupby(["q", "ratio"])[["spearman_tail_vs_centroid", "jaccard_vs_centroid_subset"]].mean()
          .to_string(float_format="%.3f"))
    regress = vs["centroid_subset_matches_committed"].dropna()
    print(f"Regression: centroid subsets equal committed files: {int(regress.sum())}/{len(regress)} checked")

    # Vocabulary sensitivity: K, sample size, sample seed.
    base_params = dict(base_spec.params)
    vocab_specs = {base_spec.tag(): base_params}
    sens = []
    for ov in args.alt_vocabs:
        key, val = parse_override(ov)
        params = {**base_params, key: val}
        spec = EmbeddingSpec(**{**base_spec.__dict__, "params": params})
        try:
            D_alt = load_embedding_matrix(ds_name, ids, spec, root=args.features_root)
        except LookupError:
            print(f"\n[skip {ov}] not derived yet:\n  python scripts/00b_derive_embeddings.py --dataset {ds_name} "
                  f"--features {spec.backbone} --method {spec.method} --k {params['K']} "
                  f"--n-sample {params['n_sample']} --seed {params['seed']}")
            continue
        vocab_specs[spec.tag()] = params
        for q, t in tails.items():
            t_alt = tail_mean(D_alt, q)
            jac = [jaccard(picks[(base_spec.tag(), q, r, seed)],
                           select_by_score(t_alt, clusters[seed].cluster, p.k, b, keep=p.keep, balance=p.balance).selected)
                   for r, b in budgets.items() for seed in args.seeds]
            sens.append({"vocab": ov, "q": q, "spearman_vs_base": spearmanr(t, t_alt)[0],
                         "mean_jaccard_vs_base": float(np.mean(jac))})
    if sens:
        sd = pd.DataFrame(sens)
        sd.to_csv(out / f"{stem}_vocab_sensitivity.csv", index=False)
        print(f"\nVocabulary sensitivity vs {base_spec.tag()} ({out / f'{stem}_vocab_sensitivity.csv'}):")
        print(sd.to_string(index=False, float_format="%.3f"))

    # Vocabulary health: cluster sizes of every fitted vocabulary used above.
    health = []
    for tag, params in vocab_specs.items():
        path = VocabSpec(base_spec.backbone, **params).path(ds_name, args.features_root)
        if not path.exists():
            continue
        v = load_vocabulary(path)
        health.append({"vocab": tag, "K": params["K"], "min_size": int(v.sizes.min()),
                       "median_size": float(np.median(v.sizes)), "max_size": int(v.sizes.max()),
                       "n_tiny(<10% of mean)": int((v.sizes < 0.1 * v.sizes.mean()).sum()),
                       "mean_cos_distance": round(v.objective, 4)})
    hd = pd.DataFrame(health)
    hd.to_csv(out / f"{ds_name}_p1_vocab_health.csv", index=False)
    print(f"\nVocabulary health ({out / f'{ds_name}_p1_vocab_health.csv'}):\n{hd.to_string(index=False)}")


if __name__ == "__main__":
    main()
