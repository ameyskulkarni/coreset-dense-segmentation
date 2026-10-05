"""Prototypicality data pruning (Sorscher et al., "Beyond neural scaling laws", NeurIPS 2022,
§6) — label-free: spherical (cosine) k-means on self-supervised image embeddings, score =
cosine distance to the nearest centroid, keep the hardest (or easiest) `budget` images.

    scoring.py   embeddings -> spherical k-means -> per-image distance + cluster   (paper steps 1-3)
    pruning.py   scores -> exactly `budget` kept indices, optional cluster balancing (step 4)
    selector.py  `select()` registry entry point + `run()` for full debug output

Self-contained: registered by one line in `../registry.py`; delete that line (and this
package) to unplug it. Walkthrough: docs/prototypicality.md.
"""
from .pruning import PruneResult, select_by_score
from .scoring import PrototypeScores, compute_prototype_scores
from .selector import PrototypicalityConfig, PrototypicalityResult, auto_name, run, select

__all__ = [
    "PrototypeScores", "compute_prototype_scores", "PruneResult", "select_by_score",
    "PrototypicalityConfig", "PrototypicalityResult", "auto_name", "run", "select",
]
