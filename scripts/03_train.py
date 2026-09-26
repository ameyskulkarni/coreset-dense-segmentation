#!/usr/bin/env python
"""Train one segmentation model on one subset with the frozen recipe (§4, §12.4, ADR 0001
— docs/adr/0001-epoch-based-training.md). Runs a fast periodic eval during training and the
full eval protocol at the end, then appends one row to results/runs.csv. Refuses to run on
a dirty git tree unless --allow-dirty.

Usage:
    python scripts/03_train.py --dataset ade20k --model segformer_b0 --recipe ade20k_proxy \\
        --subset results/subsets/ade20k_random_0.2_0.json

    python scripts/03_train.py --experiment configs/experiment/example_ade20k_random20.yaml \\
        --subset results/subsets/ade20k_random_0.2_0.json

    # train on 100% data (no --subset) for the matched-epoch-budget reference denominator:
    python scripts/03_train.py --dataset ade20k --model segformer_b0 --recipe ade20k_proxy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segcoreset.logging.runs_store import append_run
from segcoreset.logging.wandb_logger import init_wandb
from segcoreset.selection.io import load_subset
from segcoreset.train.trainer import Trainer
from segcoreset.utils.config import build_config, config_hash, save_resolved_config
from segcoreset.utils.git_utils import check_clean_tree, get_git_commit
from segcoreset.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--subset", default=None, help="path to a subset json from 02_select.py; omit to train on 100% data")
    parser.add_argument("--method", default=None, help="selection method name for provenance, if not inferable from --subset's filename")
    parser.add_argument("--ratio", type=float, default=None, help="provenance only, if not inferable from --subset's filename")
    parser.add_argument("--seed", type=int, default=None, help="training seed (RNG init/order); overrides recipe.seed_train")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    check_clean_tree(".", args.allow_dirty)

    cfg = build_config(experiment=args.experiment, dataset=args.dataset, model=args.model, recipe=args.recipe, overrides=args.overrides)
    if "dataset" not in cfg or "model" not in cfg or "recipe" not in cfg:
        raise ValueError("Need --dataset, --model, and --recipe (directly or via --experiment).")
    if args.seed is not None:
        cfg.recipe.seed_train = args.seed

    subset_ids, method, ratio = None, "full", 1.0
    if args.subset:
        subset_ids = load_subset(args.subset)
        stem_parts = Path(args.subset).stem.split("_")  # "<dataset>_<method>_<ratio>_<seed>"
        method = args.method or (stem_parts[1] if len(stem_parts) >= 4 else "unknown")
        ratio = args.ratio if args.ratio is not None else (float(stem_parts[2]) if len(stem_parts) >= 4 else None)

    git_commit = get_git_commit(".")
    chash = config_hash(cfg)
    run_name = args.run_name or f"{cfg.dataset.name}_{cfg.model.name}_{method}_r{ratio}_s{cfg.recipe.seed_train}_{chash}"
    run_dir = Path("results/checkpoints") / run_name
    save_resolved_config(cfg, run_dir / "config.yaml")

    wandb_run = init_wandb(
        cfg, run_name=run_name, tags=[cfg.dataset.name, cfg.model.name, method],
        extra_config={
            "git_commit": git_commit, "config_hash": chash,
            "selection_method": method, "ratio": ratio,
        },
    )

    trainer = Trainer(cfg, subset_ids=subset_ids, run_dir=run_dir, wandb_run=wandb_run)
    n_images = len(trainer.train_ds)
    result = trainer.train()
    metrics = result["metrics"]

    append_run({
        "run_id": run_name,
        "git_commit": git_commit,
        "config_hash": chash,
        "wandb_run_id": getattr(wandb_run, "id", None),
        "wandb_url": getattr(wandb_run, "url", None),
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "selection_method": method,
        "representation": cfg.features.name if "features" in cfg else None,
        "ratio": ratio,
        "seed": cfg.recipe.seed_train,
        "n_images": n_images,
        "epochs": result["epochs"],
        "iterations": result["iterations"],
        "lr": cfg.recipe.lr,
        "lr_schedule": cfg.recipe.get("lr_schedule"),
        "miou": metrics["miou"],
        "rare_class_miou": metrics.get("rare_class_miou"),
        "pixel_acc": metrics["pixel_acc"],
        "boundary_f": metrics.get("boundary_f"),
        "per_class_iou_json": metrics["per_class_iou"],
        "retention_vs_full_matched": None,
        "gpu_hours_train": result["gpu_hours"],
        "selection_seconds": None,
        "notes": "",
    })

    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done: miou={metrics['miou']:.4f} rare_class_miou={metrics.get('rare_class_miou')} -> {run_dir}")


if __name__ == "__main__":
    main()
