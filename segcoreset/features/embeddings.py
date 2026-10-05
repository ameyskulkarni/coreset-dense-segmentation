"""Load one per-image embedding matrix, row-aligned with a list of image ids, for a
feature-based selector. Selectors never read feature files themselves: `02_select.py`
resolves a selection config's `embedding:` block through `load_embedding_matrix` and hands
the selector a plain `[N, D]` array (row i <-> image_ids[i]).

Two sources are supported:

    derived   one row per (image, derivation) in `results/features/<dataset>/derived.parquet`
              (`00b_derive_embeddings.py`), filtered by split/backbone/method/params.
    raw       the per-image `.pt` cache (`00_extract_features.py`) under
              `results/features/<dataset>/<backbone>/<split>/`, reading tensor `key` (`cls`).

Example `embedding:` block (in a `configs/selection/*.yaml`):

    embedding:
      source: derived
      backbone: dinov2_vitb14
      method: cls          # derived: 00b method (cls | patch_mean | kmeans_hist)
      params: {}           # derived: must match the 00b params exactly, e.g. {K: 256, seed: 0}
      split: train
      key: cls             # raw only: which tensor of the cached dict to read
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from omegaconf import OmegaConf

from .derived import DerivedStore, canonical_params
from .store import FeatureStore

log = logging.getLogger(__name__)

DEFAULT_FEATURES_ROOT = Path("results/features")
SOURCES = ("derived", "raw")


@dataclass(frozen=True)
class EmbeddingSpec:
    backbone: str
    source: str = "derived"
    method: str = "cls"
    params: dict = field(default_factory=dict)
    split: str = "train"
    key: str = "cls"

    @classmethod
    def from_cfg(cls, cfg) -> "EmbeddingSpec":
        """Build and validate a spec from a selection config's `embedding:` block.

        Args:
            cfg: A `DictConfig` or plain mapping with keys among `backbone` (required),
                `source`, `method`, `params`, `split`, `key`. A null `params` becomes `{}`.

        Returns:
            The validated `EmbeddingSpec`.

        Raises:
            ValueError: On unknown keys, a missing `backbone`, or a `source` not in `SOURCES`.
        """
        d = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        unknown = set(d) - {"backbone", "source", "method", "params", "split", "key"}
        if unknown:
            raise ValueError(f"Unknown keys in embedding spec: {sorted(unknown)}")
        if "backbone" not in d:
            raise ValueError("embedding spec needs `backbone` (e.g. dinov2_vitb14)")
        spec = cls(**{**d, "params": d.get("params") or {}})
        if spec.source not in SOURCES:
            raise ValueError(f"embedding.source must be one of {SOURCES}, got '{spec.source}'")
        return spec

    def tag(self) -> str:
        """Short, filename-safe id of this embedding (no `_`, parts joined by `-`).

        Examples: `dinov2vitb14-cls`, `dinov2vitb14-patchmean`,
        `dinov2vitb14-kmeanshist-K256-seed0` (derived params, sorted), `raw-dinov2vitb14-cls`
        (raw source + tensor key); a non-train split is appended (`...-val`). Used in
        auto-generated selection names, so it is part of subset filenames.
        """
        def clean(x) -> str:
            return str(x).replace("_", "")

        if self.source == "raw":
            parts = ["raw", clean(self.backbone), clean(self.key)]
        else:
            parts = [clean(self.backbone), clean(self.method)] + [f"{clean(k)}{clean(v)}" for k, v in sorted(self.params.items())]
        if self.split != "train":
            parts.append(clean(self.split))
        return "-".join(parts)

    def describe(self) -> str:
        """Return a short human-readable label for logs and error messages.

        E.g. `derived[dinov2_vitb14/train/kmeans_hist {"K": 256, "seed": 0}]` or
        `raw[dinov2_vitb14/train/cls]`.
        """
        if self.source == "derived":
            return f"derived[{self.backbone}/{self.split}/{self.method} {canonical_params(self.params)}]"
        return f"raw[{self.backbone}/{self.split}/{self.key}]"


def load_embedding_matrix(
    dataset: str, image_ids: Sequence[str], spec: EmbeddingSpec, root: Path | str = DEFAULT_FEATURES_ROOT,
) -> np.ndarray:
    """Return a float32 `[len(image_ids), D]` matrix whose row i is image_ids[i]'s embedding.

    Fails loudly (rather than silently dropping images) if any requested id has no
    embedding, if an id is ambiguous, or if any value is non-finite.

    Args:
        dataset: Dataset name (subdirectory of `root`).
        image_ids: Ids to load, in the desired row order.
        spec: Which embedding to load (source, backbone, method/params or raw key, split).
        root: Features root directory.

    Returns:
        The `[N, D]` float32 embedding matrix.

    Raises:
        FileNotFoundError: `derived` source and no `derived.parquet` exists.
        LookupError: No matching derivation, or some ids have no embedding.
        KeyError: `raw` source and a cached file lacks the `spec.key` tensor.
        ValueError: Duplicate ids for the spec, a non-1-D raw tensor, or NaN/Inf values.
    """
    ids = list(image_ids)
    if spec.source == "derived":
        X = _load_derived(dataset, ids, spec, Path(root))
    else:
        X = _load_raw(dataset, ids, spec, Path(root))
    if not np.isfinite(X).all():
        bad = np.flatnonzero(~np.isfinite(X).all(axis=1))
        raise ValueError(f"{len(bad)} embeddings contain NaN/Inf, e.g. {[ids[i] for i in bad[:3]]}")
    log.info("Loaded %d x %d embeddings from %s", *X.shape, spec.describe())
    return X


def _load_derived(dataset: str, ids: list[str], spec: EmbeddingSpec, root: Path) -> np.ndarray:
    """Load embeddings for `ids` from the derived Parquet table, filtered by `spec`.

    Rows must match `split`, `backbone`, `method`, and canonical `params` exactly.

    Args:
        dataset: Dataset name.
        ids: Image ids, in output row order.
        spec: Embedding spec with `source == "derived"`.
        root: Features root directory.

    Returns:
        `[len(ids), D]` float32 matrix.

    Raises:
        FileNotFoundError: If the Parquet file does not exist (message includes the
            `00b_derive_embeddings.py` command to create it).
        LookupError: If no rows match `spec` (message lists available derivations) or
            some ids are missing.
        ValueError: If an id appears more than once for this spec.
    """
    store = DerivedStore(dataset, root=root)
    if not store.path.exists():
        raise FileNotFoundError(
            f"No derived embeddings at {store.path}. Create them with:\n"
            f"  python scripts/00b_derive_embeddings.py --dataset {dataset} --features {spec.backbone} --method {spec.method}"
        )
    df = store.load()
    sel = df[(df["split"] == spec.split) & (df["backbone"] == spec.backbone)
             & (df["method"] == spec.method) & (df["params"] == canonical_params(spec.params))]
    if sel.empty:
        available = df[["backbone", "split", "method", "params"]].drop_duplicates().to_string(index=False)
        raise LookupError(f"No rows in {store.path} match {spec.describe()}. Available derivations:\n{available}")
    dup = sel["image_id"].duplicated()
    if dup.any():
        raise ValueError(f"{int(dup.sum())} image ids appear more than once for {spec.describe()}, "
                         f"e.g. {sel.loc[dup, 'image_id'].head(3).tolist()}")
    by_id = sel.set_index("image_id")["embedding"]
    _check_coverage(ids, by_id.index, spec)
    return DerivedStore.to_matrix(by_id.loc[ids].to_frame())


def _load_raw(dataset: str, ids: list[str], spec: EmbeddingSpec, root: Path) -> np.ndarray:
    """Load embeddings for `ids` from the per-image `.pt` cache, reading tensor `spec.key`.

    Args:
        dataset: Dataset name.
        ids: Image ids, in output row order.
        spec: Embedding spec with `source == "raw"`.
        root: Features root; files are read from `<root>/<dataset>/<backbone>/<split>/`.

    Returns:
        `[len(ids), D]` float32 matrix.

    Raises:
        LookupError: If some ids are not cached (message includes the extraction command).
        KeyError: If a cached file has no `spec.key` tensor.
        ValueError: If that tensor is not 1-D (e.g. `patch`; pool it via `00b` instead).
    """
    raw_dir = root / dataset / spec.backbone / spec.split
    store = FeatureStore(raw_dir)
    _check_coverage(ids, store.cached_ids(), spec, hint=(
        f"  python scripts/00_extract_features.py --dataset {dataset} --features {spec.backbone} --split {spec.split}"))
    rows = []
    for i in ids:
        feats = store.load(i)
        if spec.key not in feats:
            raise KeyError(f"Cached features for '{i}' have no '{spec.key}' tensor (have {list(feats)})")
        t = feats[spec.key].float()
        if t.ndim != 1:
            raise ValueError(f"raw source needs a 1-D per-image tensor; '{spec.key}' has shape {tuple(t.shape)}. "
                             "Pool it first with 00b_derive_embeddings.py and use source: derived.")
        rows.append(t.numpy())
    return np.stack(rows).astype(np.float32, copy=False)


def _check_coverage(ids: list[str], available, spec: EmbeddingSpec, hint: str | None = None) -> None:
    """Ensure every requested id has an embedding available.

    Args:
        ids: Requested image ids.
        available: Ids that have an embedding (any iterable / index).
        spec: The spec being loaded (for the error message).
        hint: Optional command line to append to the error showing how to fill the gap.

    Raises:
        LookupError: If any id in `ids` is not in `available`.
    """
    missing = sorted(set(ids) - set(available))
    if missing:
        msg = (f"{len(missing)}/{len(ids)} image ids have no {spec.describe()} embedding, e.g. {missing[:3]}. "
               "Every candidate image must be embedded — selection never silently skips images.")
        raise LookupError(msg + (f" Extract them with:\n{hint}" if hint else ""))
