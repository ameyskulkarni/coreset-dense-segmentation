#!/usr/bin/env python
"""Compute and freeze the bottom-K rare-class list for a dataset, from FULL train-set pixel
frequency (§2). Run once per dataset; every downstream eval reports rare-class mIoU
against this frozen list. Labels are used here for analysis only, never for selection.

Usage:
    python scripts/01_compute_rare_classes.py --dataset ade20k --k 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segcoreset.data.rare_classes import compute_and_cache_rare_classes
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=None, help="defaults to the dataset config's rare_class_k")
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(dataset=args.dataset)
    k = args.k or cfg.dataset.rare_class_k
    result = compute_and_cache_rare_classes(cfg.dataset, k)

    print(f"Rare classes for {cfg.dataset.name} (bottom {k} by pixel frequency):")
    names = result["rare_class_names"] or [""] * k
    for cid, name in zip(result["rare_class_ids"], names):
        print(f"  class {cid:3d} ({name}): {result['pixel_freq'][cid] * 100:.4f}% of labeled pixels")
    print(f"Cached -> {cfg.dataset.rare_classes_cache}")


if __name__ == "__main__":
    main()
