#!/usr/bin/env python
"""Extract and cache image representations (§6): DINOv2/DINOv3 patch+CLS tokens, plus the
other representation-study arms (rn50_sup, segb0_enc, clip_cls). Resumable — already-cached
image ids are skipped.

Usage:
    python scripts/00_extract_features.py --dataset ade20k --features dinov2_vits14 --split train
    python scripts/00_extract_features.py --experiment configs/experiment/example_ade20k_random20.yaml --features dinov2_vits14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from segcoreset.data.registry import build_dataset
from segcoreset.features.registry import build_extractor
from segcoreset.features.store import FeatureStore
from segcoreset.utils.config import build_config
from segcoreset.utils.logging_setup import setup_logging


class _ImageOnlyDataset(Dataset):
    """Loads only images (no labels) for extraction speed at ADE20K/Cityscapes scale."""

    def __init__(self, samples, preprocess):
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return self.preprocess(Image.open(s.image_path).convert("RGB")), s.image_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--features", default=None, help="representation config name, e.g. dinov2_vits14")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = build_config(experiment=args.experiment, dataset=args.dataset, features=args.features, overrides=args.overrides)
    if "dataset" not in cfg or "features" not in cfg:
        raise ValueError("Need both a --dataset and a --features config (directly or via --experiment).")

    split_name = cfg.dataset.train_split if args.split == "train" else cfg.dataset.val_split
    ds = build_dataset(cfg.dataset, split=split_name)
    extractor = build_extractor(cfg.features)

    out_dir = Path("results/features") / cfg.dataset.name / cfg.features.name / args.split
    store = FeatureStore(out_dir)
    cached = store.cached_ids()
    todo = [s for s in ds.samples if s.image_id not in cached]
    print(f"{len(cached)} cached, {len(todo)} to extract -> {out_dir}")

    if todo:
        loader = DataLoader(
            _ImageOnlyDataset(todo, extractor.preprocess),
            batch_size=cfg.features.batch_size, num_workers=args.num_workers, shuffle=False,
        )
        for batch, ids in tqdm(loader, desc=f"extracting {cfg.features.name}"):
            feats = extractor.extract_batch(batch)
            for i, image_id in enumerate(ids):
                store.save(image_id, {k: v[i] for k, v in feats.items()})

    all_ids = [s.image_id for s in ds.samples]
    store.write_index(all_ids, {"dataset": cfg.dataset.name, "representation": cfg.features.name, "split": args.split})
    print(f"Done. {len(all_ids)} images cached at {out_dir}")


if __name__ == "__main__":
    main()
