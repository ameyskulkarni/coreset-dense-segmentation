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
    """Return the full SHA of `HEAD` in `repo_root`.

    Args:
        repo_root: Any path inside the git repository.

    Returns:
        The 40-character commit hash, or `"unknown"` if git is not installed or
        `repo_root` is not a git repository (never raises).
    """
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_dirty(repo_root: Path | str = ".") -> bool:
    """Report whether the working tree has uncommitted changes.

    Uses `git status --porcelain`, so untracked (non-ignored) files count as dirty too.
    Gitignored paths (e.g. `results/metrics/`, `scripts/experiments/`) never do.

    Args:
        repo_root: Any path inside the git repository.

    Returns:
        `True` if there are modified, staged, or untracked files; `False` if the tree is
        clean OR if git is unavailable / this is not a repository (fails open).
    """
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return len(out.strip()) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_clean_tree(repo_root: Path | str, allow_dirty: bool) -> None:
    """Refuse to proceed on a dirty working tree unless explicitly allowed.

    This gate is what makes the `git_commit` column in `runs.csv` trustworthy: a run logged
    against a commit must have actually run that commit's code and inputs.

    Args:
        repo_root: Any path inside the git repository.
        allow_dirty: If `True`, skip the check entirely (for quick smoke tests).

    Raises:
        RuntimeError: If `allow_dirty` is `False` and `is_dirty(repo_root)` is `True`.
    """
    if not allow_dirty and is_dirty(repo_root):
        raise RuntimeError(
            "Git working tree is dirty. Commit or stash before launching a run (this is "
            "what makes the git_commit column in runs.csv trustworthy), or pass --allow-dirty "
            "to override for a quick smoke test."
        )
