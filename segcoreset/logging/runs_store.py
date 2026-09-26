"""The append-only results ledger (§12.3) — the whole paper is aggregated from this file.
One row per finished run; never overwrite. `retention_vs_full_matched` is left blank at
write time (it needs the matching full@matched-budget run to exist first) and is meant to
be backfilled by a future aggregation script.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

RUNS_CSV_COLUMNS = [
    "run_id", "git_commit", "config_hash", "wandb_run_id", "wandb_url", "timestamp",
    "dataset", "model", "selection_method", "representation",
    "ratio", "seed", "n_images", "epochs", "iterations", "lr", "lr_schedule",
    "miou", "rare_class_miou", "pixel_acc", "boundary_f",
    "per_class_iou_json",
    "retention_vs_full_matched", "gpu_hours_train", "selection_seconds",
    "notes",
]

DEFAULT_RUNS_CSV = Path("results/runs.csv")


def append_run(row: dict, csv_path: Path | str = DEFAULT_RUNS_CSV) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    if isinstance(row.get("per_class_iou_json"), (dict, list)):
        row["per_class_iou_json"] = json.dumps(row["per_class_iou_json"])

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RUNS_CSV_COLUMNS})
