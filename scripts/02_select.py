#!/usr/bin/env python
"""Select an image subset with a label-free method and cache it (§12.2, §12.4). Selection
happens before any training, using cached features (or none, for `random`/`full`).

Usage:
    python scripts/02_select.py --dataset ade20k --selection random --ratio 0.2 --seed 0
    python scripts/02_select.py --experiment configs/experiment/example_ade20k_random20.yaml --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segcoreset.data.registry import build_dataset
from segcoreset.selection.io import save_subset, subset_path
from segcoreset.selection.registry import get_selector
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--ratio", type=float, default=None, help="fraction of train images to keep")
    parser.add_argument("--seed", type=int, default=0)
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

    selector = get_selector(cfg.selection.method)
    chosen = selector(image_ids, None, budget, args.seed, cfg.selection)

    ratio_for_name = 1.0 if cfg.selection.method == "full" else ratio
    out_path = subset_path(cfg.selection.method, cfg.dataset.name, ratio_for_name, args.seed)
    save_subset(out_path, chosen, {
        "dataset": cfg.dataset.name, "method": cfg.selection.method,
        "ratio": ratio_for_name, "seed": args.seed, "n_images": len(chosen), "n_total": len(image_ids),
    })
    print(f"Selected {len(chosen)}/{len(image_ids)} images -> {out_path}")


if __name__ == "__main__":
    main()
