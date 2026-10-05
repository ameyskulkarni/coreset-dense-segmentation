"""Thin W&B wrapper — the rest of the codebase never imports `wandb` directly, so swapping
or disabling the logging backend touches only this file."""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig, run_name: str, tags: list[str] | None = None,
               extra_config: dict | None = None):
    """Start a W&B run with the full resolved config attached.

    Respects `cfg.wandb.mode` (`online`/`offline`/`disabled`, default `online`). Note that
    `wandb` is imported unconditionally, so it must be installed even when mode is
    `disabled`; in that mode `wandb.init` returns a no-op run object rather than `None`.

    `extra_config` is merged into the logged config on top of `cfg` — use it for
    provenance that lives outside `cfg` itself (git commit, config hash, the resolved
    selection method/ratio when they were parsed from a subset filename rather than
    present as literal fields in `cfg`) so every run is uniformly filterable in the W&B
    UI regardless of whether it was launched via `--experiment` or bare CLI flags.

    Args:
        cfg: Full resolved run config; optional `wandb:` block with `project`, `entity`,
            and `mode`.
        run_name: Display name of the run.
        tags: Optional W&B tags.
        extra_config: Extra top-level config keys (override same-named `cfg` keys).

    Returns:
        The `wandb.Run` returned by `wandb.init`.
    """
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
