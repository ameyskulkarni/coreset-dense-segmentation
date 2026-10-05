"""Derived-embedding table: one Parquet file per dataset holding one row per
(image, derivation). The raw `cls`/`patch` cache (`store.py`) is never modified — these
are cheap, recomputable manipulations of it (patch mean-pooling, k-means histograms, ...).

Embeddings are stored as float32 lists (lossless for float32 inputs; Parquet is binary, so
no text truncation). Provenance columns say how each vector was made, so methods/params
can be filtered and compared side by side.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# (image_id, backbone, source, method, params) identifies one derived embedding.
KEY_COLS = ["image_id", "split", "backbone", "source", "method", "params"]
_SCHEMA = pa.schema([
    ("image_id", pa.string()),
    ("image_path", pa.string()),  # source image on disk (from the raw cache's index.json)
    ("dataset", pa.string()),
    ("split", pa.string()),
    ("backbone", pa.string()),  # e.g. dinov2_vitb14
    ("source", pa.string()),  # raw tensor it was derived from: cls | patch
    ("method", pa.string()),  # cls | patch_mean | kmeans_hist | ...
    ("params", pa.string()),  # canonical JSON, e.g. {"K": 256, "seed": 0}
    ("dim", pa.int32()),
    ("embedding", pa.list_(pa.float32())),
    ("config_hash", pa.string()),
    ("created_at", pa.string()),
])


def canonical_params(params: dict | None) -> str:
    """Serialize derivation params to a canonical string for storage and exact-match lookup.

    Keys are sorted so `{"seed": 0, "K": 256}` and `{"K": 256, "seed": 0}` compare equal.

    Args:
        params: Derivation parameters, or `None` (treated as `{}`).

    Returns:
        A JSON string, e.g. `'{"K": 256, "seed": 0}'` or `'{}'`.
    """
    return json.dumps(params or {}, sort_keys=True)


class DerivedStore:
    def __init__(self, dataset: str, root: Path | str = "results/features"):
        """Point the store at `<root>/<dataset>/derived.parquet` (nothing is read or created yet).

        Args:
            dataset: Dataset name (e.g. `cityscapes`).
            root: Features root directory, relative to the CWD by default.
        """
        self.dataset = dataset
        self.path = Path(root) / dataset / "derived.parquet"

    def load(self) -> pd.DataFrame:
        """Read the whole derived-embedding table.

        Returns:
            A DataFrame with the `_SCHEMA` columns (`embedding` holds float32 arrays), or an
            empty DataFrame with those columns if the file does not exist yet.
        """
        if not self.path.exists():
            return pd.DataFrame(columns=[f.name for f in _SCHEMA])
        return pq.read_table(self.path).to_pandas()

    def add(
        self, image_ids, embeddings: np.ndarray, *, split: str, backbone: str, source: str,
        method: str, params: dict | None = None, config_hash: str = "", image_paths=None,
    ) -> None:
        """Append (or replace, if the same key already exists) one derivation for many images.

        Rows are keyed by `KEY_COLS` (image_id, split, backbone, source, method, params):
        existing rows with a matching key are dropped and the new ones appended, so re-running
        a derivation is idempotent. The whole Parquet file is rewritten on every call.

        Args:
            image_ids: Image ids, one per row of `embeddings`.
            embeddings: `[N, D]` array (cast to contiguous float32).
            split: Split the embeddings were computed on (e.g. `train`).
            backbone: Raw feature config name (e.g. `dinov2_vitb14`).
            source: Raw tensor derived from: `cls` or `patch`.
            method: Derivation method (e.g. `cls`, `patch_mean`, `kmeans_hist`).
            params: Method parameters; stored via `canonical_params`.
            config_hash: Provenance hash of the config that produced the embeddings.
            image_paths: Optional source image paths, one per id (empty string if omitted).

        Raises:
            ValueError: If `embeddings` is not 2-D or its row count differs from `image_ids`.
        """
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or len(emb) != len(image_ids):
            raise ValueError(f"embeddings must be [N, D] matching image_ids; got {emb.shape} for {len(image_ids)} ids")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new = pd.DataFrame({
            "image_id": list(image_ids), "image_path": list(image_paths) if image_paths is not None else "",
            "dataset": self.dataset, "split": split,
            "backbone": backbone, "source": source, "method": method,
            "params": canonical_params(params), "dim": np.int32(emb.shape[1]),
            "embedding": list(emb), "config_hash": config_hash, "created_at": now,
        })
        old = self.load()
        if len(old):
            keys = pd.MultiIndex.from_frame(new[KEY_COLS])
            old = old[~pd.MultiIndex.from_frame(old[KEY_COLS]).isin(keys)]
        df = pd.concat([old, new], ignore_index=True) if len(old) else new
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False), self.path)

    @staticmethod
    def to_matrix(df: pd.DataFrame) -> np.ndarray:
        """Stack a filtered frame's `embedding` column into an [N, D] float32 array.

        Args:
            df: Rows to stack (in their current order); all embeddings must share one length.

        Returns:
            `[len(df), D]` float32 array, row i = `df.iloc[i]`'s embedding.
        """
        return np.stack(df["embedding"].to_numpy()).astype(np.float32, copy=False)
