#!/usr/bin/env python
"""Select an image subset with a label-free method and cache it (§12.2, §12.4). Selection
happens before any training, using cached features (or none, for `random`/`full`).

Usage:
    python scripts/02_select.py --dataset ade20k --selection random --ratio 0.2 --seed 0
    python scripts/02_select.py --experiment configs/experiment/example_ade20k_random20.yaml --seed 0
    python scripts/02_select.py --dataset cityscapes --selection prototypicality-hard --ratio 0.2 --seed 0

Feature-based selectors declare an `embedding:` block in their selection config; the matching
saved embeddings are loaded (segcoreset/features/embeddings.py) and passed to the selector as
an [N, D] matrix row-aligned with the image ids. The subset filename uses `selection.name`
(`name: auto` generates a descriptive one, e.g. `proto-hard-k20-bal50-dinov2vitb14-cls`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from segcoreset.data.registry import build_dataset
from segcoreset.features.embeddings import DEFAULT_FEATURES_ROOT, EmbeddingSpec, load_embedding_matrix
from segcoreset.selection.io import load_subset, save_subset, subset_path
from segcoreset.selection.registry import get_selector, selection_name
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging


def main():
    """CLI entry point: run one selector over the full train split and write the subset file.

    Budget is `round(ratio * N)` (all N for `full`). If the selection config has an
    `embedding:` block, the matching embeddings are loaded and passed as `features`. An
    existing subset file with identical ids is kept untouched; one with different ids is
    only overwritten with `--force`.

    Raises:
        ValueError: If dataset/selection config is missing, `--ratio` is missing for a
            non-`full` method, or `selection.name` contains `_`.
        SystemExit: If the output exists with different ids and `--force` was not given.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--ratio", type=float, default=None, help="fraction of train images to keep")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing subset file even if its image ids would change")
    parser.add_argument("--features-root", default=str(DEFAULT_FEATURES_ROOT),
                        help="where saved embeddings live (feature-based selectors only)")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(experiment=args.experiment, dataset=args.dataset, selection=args.selection, overrides=args.overrides)
    if "dataset" not in cfg or "selection" not in cfg:
        raise ValueError("Need both a --dataset and a --selection config (directly or via --experiment).")

    ratio = args.ratio if args.ratio is not None else cfg.get("ratio")
    if ratio is None and cfg.selection.method != "full":
        raise ValueError("Need --ratio (or `ratio:` in the experiment/override config).")

    ds = build_dataset(cfg.dataset, split=cfg.dataset.train_split)
    image_ids = ds.image_ids()
    budget = len(image_ids) if cfg.selection.method == "full" else round(ratio * len(image_ids))

    name = selection_name(cfg.selection)  # `name: auto` -> generated from the settings

    features = None
    if "embedding" in cfg.selection:
        spec = EmbeddingSpec.from_cfg(cfg.selection.embedding)
        features = load_embedding_matrix(cfg.dataset.name, image_ids, spec, root=args.features_root)

    selector = get_selector(cfg.selection.method)
    t0 = time.perf_counter()
    chosen = selector(image_ids, features, budget, args.seed, cfg.selection)
    selection_seconds = round(time.perf_counter() - t0, 2)

    ratio_for_name = 1.0 if cfg.selection.method == "full" else ratio
    out_path = subset_path(name, cfg.dataset.name, ratio_for_name, args.seed)
    if out_path.exists():
        if load_subset(out_path) == chosen:
            print(f"Subset unchanged, keeping existing file: {out_path}")
            return
        if not args.force:
            raise SystemExit(f"Refusing to overwrite {out_path}: the selected image ids differ from the existing file "
                             f"(selector/config changed?). Pass --force to overwrite deliberately.")
    save_subset(out_path, chosen, {
        "dataset": cfg.dataset.name, "method": cfg.selection.method,
        "ratio": ratio_for_name, "seed": args.seed, "n_images": len(chosen), "n_total": len(image_ids),
        "name": name, "selection_cfg": OmegaConf.to_container(cfg.selection, resolve=True),
        "selection_seconds": selection_seconds,
    })
    print(f"Selected {len(chosen)}/{len(image_ids)} images -> {out_path}")


if __name__ == "__main__":
    main()
