"""Thin W&B wrapper — the rest of the codebase never imports `wandb` directly, so swapping
or disabling the logging backend touches only this file."""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig, run_name: str, tags: list[str] | None = None,
               extra_config: dict | None = None):
    """Respects `cfg.wandb.mode` (`online`/`offline`/`disabled`, default `online`). Returns
    `None` (and disables logging silently downstream) if `wandb` isn't installed/configured
    and mode is `disabled`.

    `extra_config` is merged into the logged config on top of `cfg` — use it for
    provenance that lives outside `cfg` itself (git commit, config hash, the resolved
    selection method/ratio when they were parsed from a subset filename rather than
    present as literal fields in `cfg`) so every run is uniformly filterable in the W&B
    UI regardless of whether it was launched via `--experiment` or bare CLI flags."""
    import wandb

    wandb_cfg = cfg.get("wandb", {})
    config = OmegaConf.to_container(cfg, resolve=True)
    if extra_config:
        config.update(extra_config)
    return wandb.init(
        project=wandb_cfg.get("project", "coreset-dense-segmentation"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        tags=tags,
        mode=wandb_cfg.get("mode", "online"),
        config=config,
    )
