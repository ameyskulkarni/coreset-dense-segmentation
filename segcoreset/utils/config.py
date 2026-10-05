"""Configuration loading (§12.5).

Composes dataset/model/recipe/selection/features YAML fragments (`configs/<group>/<name>.yaml`)
into one resolved config. Three ways to drive a script, in increasing precedence:

1. `--dataset foo --model bar ...`      -- pick named fragments directly.
2. `--experiment path/to/exp.yaml`      -- a small YAML that names fragments by group
                                            (`dataset: cityscapes`) and/or sets fields
                                            directly (`ratio: 0.2`, `wandb: {...}`).
3. `--set key.path=value ...`           -- OmegaConf dotlist overrides, highest precedence.

This keeps one config file per experiment as the primary workflow, while every field
stays reachable from the CLI for quick one-off changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

_GROUPS = ("dataset", "model", "recipe", "selection", "features")


def _load_group(group: str, name: str) -> DictConfig:
    """Load one named YAML fragment from `configs/<group>/<name>.yaml`.

    Args:
        group: Config group directory, one of `_GROUPS` (e.g. `"dataset"`, `"recipe"`).
        name: Fragment basename without the `.yaml` suffix (e.g. `"cityscapes"`).

    Returns:
        The fragment as an (unresolved) `DictConfig`.

    Raises:
        FileNotFoundError: If no such fragment exists in that group.
    """
    path = CONFIGS_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config named '{name}' in group '{group}' (looked at {path})")
    return OmegaConf.load(path)


def build_config(experiment: str | None = None, overrides: Sequence[str] = (), **group_names: str | None) -> DictConfig:
    """Build a fully resolved run config from the three composable sources.

    Sources are merged in increasing precedence: the `experiment` YAML (its group-name
    entries are expanded to the named fragments first, then its remaining fields are merged
    on top), then each explicit `group_names` fragment, then the `overrides` dotlist. Each
    group's fragment lands under its own top-level key (`cfg.dataset`, `cfg.recipe`, ...).

    Args:
        experiment: Path to an experiment YAML, or the bare name of one under
            `configs/experiment/`. Group keys holding a string (e.g. `dataset: cityscapes`)
            are treated as fragment names; every other key is merged in as a direct field.
        overrides: OmegaConf dotlist entries (`"recipe.lr=1e-4"`), applied last.
        **group_names: Fragment name per group, keyed by group (e.g. `dataset="cityscapes"`).
            `None` values are ignored so scripts can pass every CLI flag unconditionally.

    Returns:
        The merged `DictConfig`. Groups that no source mentioned are simply absent, so
        callers check e.g. `"dataset" in cfg` to validate what they need.

    Raises:
        ValueError: If a `group_names` key is not one of `_GROUPS`.
        FileNotFoundError: If a named fragment does not exist.
    """
    cfg = OmegaConf.create({})

    if experiment is not None:
        exp_path = Path(experiment)
        if not exp_path.exists():
            exp_path = CONFIGS_DIR / "experiment" / f"{experiment}.yaml"
        exp_cfg = OmegaConf.load(exp_path)
        for group in _GROUPS:
            if group in exp_cfg and isinstance(exp_cfg[group], str):
                cfg = OmegaConf.merge(cfg, {group: _load_group(group, exp_cfg[group])})
        leftover = OmegaConf.create({k: v for k, v in exp_cfg.items() if not (k in _GROUPS and isinstance(exp_cfg[k], str))})
        cfg = OmegaConf.merge(cfg, leftover)

    for group, name in group_names.items():
        if name is None:
            continue
        if group not in _GROUPS:
            raise ValueError(f"'{group}' is not a config group; expected one of {_GROUPS}")
        cfg = OmegaConf.merge(cfg, {group: _load_group(group, name)})

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    return cfg


def config_hash(cfg: DictConfig) -> str:
    """Short, stable hash of the fully resolved config — provenance for `runs.csv` (§12.3).

    The config is resolved (interpolations expanded), serialized as key-sorted JSON, and
    SHA-256 hashed, so two configs hash equal iff their resolved contents are equal,
    regardless of which input method (fragments / experiment / `--set`) produced them.

    Args:
        cfg: The resolved run config.

    Returns:
        The first 12 hex characters of the SHA-256 digest.
    """
    payload = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def save_resolved_config(cfg: DictConfig, path: Path) -> None:
    """Write `cfg` to `path` as YAML, creating parent directories as needed.

    `03_train.py` saves this next to the checkpoints as `config.yaml`; `04_eval.py` reloads it
    to rebuild the exact model/dataset config for standalone re-scoring.

    Args:
        cfg: The config to save.
        path: Destination YAML file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
