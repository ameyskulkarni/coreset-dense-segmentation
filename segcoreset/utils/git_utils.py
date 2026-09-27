"""Git provenance (§12.5): every run logs the exact commit it ran at, and refuses to run
on a dirty tree unless explicitly overridden — this is what makes `runs.csv` trustworthy."""
from __future__ import annotations

import subprocess
from pathlib import Path

# results/runs.csv is a pure OUTPUT ledger: every row already records its own git_commit,
# so the file being one run ahead of HEAD doesn't compromise that row's provenance. Without
# this exclusion, finishing run N would dirty the tree and block run N+1 from even
# starting — which broke back-to-back experiment scripts (e.g. run_cityscapes_calibration.sh)
# on an otherwise clean tree. Inputs that actually affect what a run does (code, configs,
# subset files) are NOT exempted here and still gate normally.
_DIRTY_CHECK_IGNORE = ("results/runs.csv",)


def get_git_commit(repo_root: Path | str = ".") -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_dirty(repo_root: Path | str = ".", ignore: tuple[str, ...] = _DIRTY_CHECK_IGNORE) -> bool:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    changed_paths = [line[3:] for line in out.decode().splitlines() if line.strip()]
    return any(path not in ignore for path in changed_paths)


def check_clean_tree(repo_root: Path | str, allow_dirty: bool) -> None:
    if not allow_dirty and is_dirty(repo_root):
        raise RuntimeError(
            "Git working tree is dirty. Commit or stash before launching a run (this is "
            "what makes the git_commit column in runs.csv trustworthy), or pass --allow-dirty "
            "to override for a quick smoke test."
        )
