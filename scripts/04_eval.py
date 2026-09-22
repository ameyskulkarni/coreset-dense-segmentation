#!/usr/bin/env python
"""Standalone evaluation: reload a trained checkpoint and run the full eval protocol
independently of training (§12.4). Appends a NEW row to runs.csv; never mutates existing
rows. Use this to re-score a run (a different split, full boundary-F if training skipped
it, etc.) without retraining.

Usage:
    python scripts/04_eval.py --run-dir results/checkpoints/<run_name>
    python scripts/04_eval.py --run-dir results/checkpoints/<run_name> --checkpoint last.pt --max-images 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from omegaconf import OmegaConf

from segcoreset.eval.evaluator import Evaluator
from segcoreset.logging.runs_store import append_run
from segcoreset.models.registry import build_model
from segcoreset.utils.config import config_hash
from segcoreset.utils.git_utils import get_git_commit
from segcoreset.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="directory written by 03_train.py (contains config.yaml + a checkpoint)")
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--split", default=None, help="defaults to dataset.val_split")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--notes", default="standalone_eval")
    args = parser.parse_args()

    setup_logging()
    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model, num_classes=cfg.dataset.num_classes).to(device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    evaluator = Evaluator(cfg.dataset, device=device)
    metrics = evaluator.evaluate(model, split=args.split, max_images=args.max_images, compute_boundary_f=True)

    print(f"miou={metrics['miou']:.4f} rare_class_miou={metrics.get('rare_class_miou')} "
          f"pixel_acc={metrics['pixel_acc']:.4f} boundary_f={metrics.get('boundary_f')}")

    append_run({
        "run_id": f"{run_dir.name}_eval",
        "git_commit": get_git_commit("."),
        "config_hash": config_hash(cfg),
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "selection_method": None,
        "representation": cfg.features.name if "features" in cfg else None,
        "ratio": None,
        "seed": cfg.recipe.seed_train,
        "n_images": None,
        "epochs": cfg.recipe.get("epochs"),
        "iterations": ckpt.get("iteration"),
        "miou": metrics["miou"],
        "rare_class_miou": metrics.get("rare_class_miou"),
        "pixel_acc": metrics["pixel_acc"],
        "boundary_f": metrics.get("boundary_f"),
        "per_class_iou_json": metrics["per_class_iou"],
        "retention_vs_full_matched": None,
        "gpu_hours_train": None,
        "selection_seconds": None,
        "notes": args.notes,
    })


if __name__ == "__main__":
    main()
