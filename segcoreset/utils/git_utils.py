"""Git provenance (§12.5): every run logs the exact commit it ran at, and refuses to run
on a dirty tree unless explicitly overridden — this is what makes provenance trustworthy.

The results ledger (results/metrics/runs.csv) is gitignored (a mutable output, not an
input — each row already self-describes its provenance via git_commit/config_hash/
wandb_url), so it never shows up in `git status` and never needs special-casing here.
Only real inputs (code, configs, subset files) can make the tree dirty."""
from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_commit(repo_root: Path | str = ".") -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_dirty(repo_root: Path | str = ".") -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return len(out.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_clean_tree(repo_root: Path | str, allow_dirty: bool) -> None:
    if not allow_dirty and is_dirty(repo_root):
        raise RuntimeError(
            "Git working tree is dirty. Commit or stash before launching a run (this is "
            "what makes the git_commit column in runs.csv trustworthy), or pass --allow-dirty "
            "to override for a quick smoke test."
        )
