"""Checkpoint save/load."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, model, optimizer, iteration: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        **(extra or {}),
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
