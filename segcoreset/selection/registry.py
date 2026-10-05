"""Selector registry (§12.2). Implemented: `random`, `full`, `prototypicality`. The remaining
label-free baselines (k-center, SemDeDup, bpp, ZCore) and `patch_coverage` (§5) drop
in as new modules here without touching the trainer, evaluator, or scripts."""
from __future__ import annotations

from . import full_select, prototypicality, random_select

SELECTOR_REGISTRY = {
    "random": random_select.select,
    "full": full_select.select,
    "prototypicality": prototypicality.select,  # docs/prototypicality.md
}

# Methods that support `name: auto` (a name generated from every subset-changing setting).
AUTO_NAMERS = {
    "prototypicality": prototypicality.auto_name,
}


def selection_name(sel_cfg) -> str:
    """Resolve a `selection:` config's name: its `name` field (default: the method), or the
    generated name when it is `auto`.

    The name is used in the subset filename `<dataset>_<name>_<ratio>_<seed>.json` and becomes
    `selection_method` in runs.csv; `03_train.py` splits that filename on `_`, so the name must
    not contain underscores.

    Args:
        sel_cfg: The resolved `selection:` sub-config.

    Returns:
        The name.

    Raises:
        ValueError: If `name: auto` is used with a method that has no auto-namer, or the name
            contains `_`.
    """
    name = sel_cfg.get("name", sel_cfg["method"])
    if name == "auto":
        if sel_cfg["method"] not in AUTO_NAMERS:
            raise ValueError(f"selection.name: auto is not supported for method '{sel_cfg['method']}'; set a name")
        name = AUTO_NAMERS[sel_cfg["method"]](sel_cfg)
    if "_" in name:  # 03_train.py recovers method/ratio from "<dataset>_<name>_<ratio>_<seed>"
        raise ValueError(f"selection.name '{name}' must not contain '_' (use '-'); it is part of the subset filename.")
    return name


def get_selector(method: str):
    """Look up a selection method's `select` function by name.

    Args:
        method: Key in `SELECTOR_REGISTRY` (the config's `selection.method`).

    Returns:
        A callable with the `base.select` signature.

    Raises:
        ValueError: If `method` is not registered.
    """
    if method not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown selection method '{method}'. Available: {list(SELECTOR_REGISTRY)}. "
            "Add new methods as modules under segcoreset/selection/ — see base.py for the interface."
        )
    return SELECTOR_REGISTRY[method]
