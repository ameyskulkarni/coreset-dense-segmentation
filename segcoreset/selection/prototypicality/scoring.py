"""Step 1-3 of prototypicality (Sorscher et al. 2022, §6): L2-normalize the embeddings,
cluster them with spherical k-means, and score every image by its cosine distance to the
nearest centroid ("prototype"). Small distance = easy = most prototypical; large = hard.

Spherical k-means (cosine k-means) assigns points by cosine similarity and keeps every
centroid unit-norm, so clustering and scoring use one geometry: each image's cluster IS
its nearest prototype by cosine, and the objective being optimized (total cosine distance
to the assigned prototypes) is the quantity the score measures.

Pure numpy, no I/O — feed it any `[N, D]` matrix.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrototypeScores:
    """Per-image outputs of the metric, row-aligned with the input embedding matrix."""

    distance: np.ndarray  # [N] cosine distance to nearest prototype, in [0, 2]; higher = harder
    cluster: np.ndarray  # [N] index of that nearest prototype (= spherical k-means assignment)
    centroids: np.ndarray  # [k, D] unit-norm prototypes
    inertia: float  # sum ||x - c||^2 over images and their unit prototypes = 2 * sum(distance)

    @property
    def k(self) -> int:
        """Number of prototypes (k-means clusters)."""
        return len(self.centroids)

    def cluster_sizes(self) -> np.ndarray:
        """Return `[k]` counts of images assigned to each prototype."""
        return np.bincount(self.cluster, minlength=self.k)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row of `X` to unit L2 norm.

    Args:
        X: `[N, D]` matrix.
        eps: Rows with norm below this are rejected.

    Returns:
        `[N, D]` row-normalized matrix (same dtype family as `X`).

    Raises:
        ValueError: If any row has (near-)zero norm.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if (norms < eps).any():
        raise ValueError(f"{int((norms < eps).sum())} embeddings have ~zero norm; cosine distance is undefined for them")
    return X / norms


def _init_plusplus(Xn: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy k-means++ seeding on the unit sphere (the variant sklearn uses).

    Each step draws `2 + log(k)` candidate centres with probability proportional to their
    cosine distance to the nearest centre chosen so far, and keeps the candidate that most
    reduces the total. For unit vectors ||x - c||^2 = 2 (1 - cos(x, c)), so this is standard
    k-means++ (sampling proportional to squared Euclidean distance).
    """
    n = len(Xn)
    n_trials = 2 + int(np.log(k))
    idx = [int(rng.integers(n))]
    closest = np.clip(1.0 - Xn @ Xn[idx[0]], 0.0, None)
    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:  # every point already coincides with a centre
            idx.append(int(rng.integers(n)))
            continue
        cand = np.searchsorted(np.cumsum(closest), rng.random(n_trials) * total)
        cand = np.minimum(cand, n - 1)
        cand_dist = np.minimum(closest[None, :], np.clip(1.0 - Xn[cand] @ Xn.T, 0.0, None))  # [trials, N]
        best = int(cand_dist.sum(axis=1).argmin())
        idx.append(int(cand[best]))
        closest = cand_dist[best]
    return Xn[idx].copy()


def spherical_kmeans(Xn: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 300,
                     eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, bool]:
    """One restart of spherical k-means (Lloyd iterations with cosine assignment).

    Repeats: assign every point to the prototype with the highest cosine; set every prototype
    to the normalized sum of its members. Stops when assignments no longer change (a fixed
    point: the objective can't improve further) or after `max_iter` iterations. A prototype
    left with no members is re-seeded at the point currently farthest from its own prototype.

    Args:
        Xn: `[N, D]` unit-norm rows.
        k: Number of prototypes.
        rng: Random generator (used only for the k-means++ seeding).
        max_iter: Maximum assignment/update iterations.
        eps: Norm below which a prototype's member sum counts as empty.

    Returns:
        `(centroids [k, D] unit-norm, labels [N] = argmax cosine to the returned centroids,
        converged)`.
    """
    n = len(Xn)
    C = _init_plusplus(Xn, k, rng)
    labels = None
    converged = False
    for _ in range(max_iter):
        sim = Xn @ C.T
        new = sim.argmax(axis=1)
        if labels is not None and np.array_equal(new, labels):
            converged = True
            break
        labels = new
        onehot = np.zeros((k, n))
        onehot[labels, np.arange(n)] = 1.0
        S = onehot @ Xn
        norms = np.linalg.norm(S, axis=1)
        alive = norms > eps
        C[alive] = S[alive] / norms[alive, None]
        if not alive.all():
            # re-seed empty prototypes at the points farthest from their current prototype
            far = np.argsort(sim[np.arange(n), labels], kind="stable")
            dist = 1.0 - sim[far, labels[far]]
            reseeded = list(zip(np.flatnonzero(~alive), far[dist > eps]))
            for j, i in reseeded:
                C[j] = Xn[i]
            if reseeded:  # assignments must be recomputed before convergence can be declared
                labels = None
            # (if every point already sits on a prototype, e.g. fewer distinct points than k,
            # the empty prototypes simply stay empty)
    sim = Xn @ C.T
    return C, sim.argmax(axis=1), converged


def compute_prototype_scores(X: np.ndarray, k: int, seed: int, n_init: int = 10, max_iter: int = 300) -> PrototypeScores:
    """Cluster `X` into `k` prototypes with spherical k-means; score each row by cosine distance
    to its prototype.

    Runs `n_init` restarts (k-means++ seeding each), keeps the one with the lowest total cosine
    distance (ties: the earliest), and scores `distance = 1 - cos(x, prototype)`. Because the
    prototypes are unit-norm and assignment is by cosine, every image's cluster is exactly its
    nearest prototype.

    Args:
        X: `[N, D]` embeddings (any float dtype; computed in float64).
        k: Number of prototypes, in [1, N].
        seed: Seeds the k-means++ draws of all restarts (the only stochastic step).
        n_init: Restarts; the lowest-objective one is kept.
        max_iter: Max assignment/update iterations per restart.

    Returns:
        A `PrototypeScores` row-aligned with `X`.

    Raises:
        ValueError: If `X` is not 2-D, `k` is out of range, or a row has ~zero norm.
    """
    X = np.asarray(X, dtype=np.float64)  # float64: float32 sums drift with thread summation order
    if X.ndim != 2:
        raise ValueError(f"expected an [N, D] embedding matrix, got shape {X.shape}")
    n = len(X)
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must be in [1, N={n}]")

    Xn = l2_normalize(X)
    rng = np.random.default_rng(seed)
    best = None
    n_unconverged = 0
    for _ in range(n_init):
        C, labels, converged = spherical_kmeans(Xn, k, rng, max_iter=max_iter)
        n_unconverged += not converged
        objective = float((1.0 - (Xn * C[labels]).sum(axis=1)).sum())
        if best is None or objective < best[0]:
            best = (objective, C, labels)
    _, C, cluster = best
    if n_unconverged:
        log.warning("%d/%d spherical k-means restarts hit max_iter=%d before converging", n_unconverged, n_init, max_iter)

    distance = np.clip(1.0 - (Xn * C[cluster]).sum(axis=1), 0.0, 2.0)
    empty = int((np.bincount(cluster, minlength=k) == 0).sum())
    if empty:
        log.warning("%d/%d prototypes have no members (k may be too large for N=%d)", empty, k, n)

    return PrototypeScores(distance=distance, cluster=cluster, centroids=C, inertia=float(2.0 * distance.sum()))
