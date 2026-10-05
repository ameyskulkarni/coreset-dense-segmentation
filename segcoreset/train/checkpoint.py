"""Checkpoint save/load."""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, model, optimizer, iteration: int, extra: dict | None = None) -> None:
    """Save model + optimizer state and the step counter to `path` (parents created).

    Args:
        path: Destination `.pt` file (e.g. `<run_dir>/last.pt` or `final.pt`).
        model: Module whose `state_dict()` is stored under `"model"`.
        optimizer: Optimizer whose `state_dict()` is stored under `"optimizer"`.
        iteration: Optimizer steps completed, stored under `"iteration"`.
        extra: Optional additional top-level keys merged into the saved dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        **(extra or {}),
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, map_location="cpu") -> dict:
    """Load a checkpoint written by `save_checkpoint` into `model` (and `optimizer`).

    Args:
        path: The `.pt` file.
        model: Module to load `"model"` weights into (strict).
        optimizer: If given and the checkpoint has `"optimizer"`, its state is restored.
        map_location: Passed to `torch.load`.

    Returns:
        The full checkpoint dict (e.g. to read `"iteration"`).
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
