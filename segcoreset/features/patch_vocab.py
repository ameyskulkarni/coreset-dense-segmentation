"""Patch vocabulary: spherical k-means over a fixed random sample of patch tokens, shared by
every patch-level derivation (`00b_derive_embeddings.py`: `kmeans_hist`, `patch_dist`) and by
`patch_coverage` (Proposal 2). Walkthrough: docs/proposals-explained.md, "Proposal 1".

    sample_patches     n patches per image, uniform without replacement, seeded per image id
    fit_vocabulary     spherical k-means (k-means++ seeding, several restarts), torch, float32
    save/load          results/features/<dataset>/vocab/<tag>.npz (centroids + cluster sizes)
    assign_patches     one image's full patch grid -> nearest-prototype index + cosine distance

Why a sample, not every patch: all Cityscapes train patches are ~4.1M x 768 (~12.5 GB float32).
200 per image is ~595k vectors (~1.8 GB), drawn from every image, shuffled by construction.
This replaces the old streaming `MiniBatchKMeans.partial_fit` fit (audit #4), whose centroids
were all seeded from the first image and dominated by the first cities in sorted order.

The vocabulary is fitted once on the train split and saved; derivations load the saved
centroids, so reruns are reproducible even though GPU matmuls are not bit-deterministic.
"""
from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from .store import FeatureStore

log = logging.getLogger(__name__)

N_INIT = 3  # k-means restarts; fixed (not a derivation param) so vocab names stay short
MAX_ITER = 300


@dataclass(frozen=True)
class VocabSpec:
    """Everything that changes a fitted vocabulary (the derivation `params` of `kmeans_hist`
    and `patch_dist`)."""

    backbone: str
    K: int = 256
    n_sample: int = 200  # patches sampled per train image for the fit
    seed: int = 0

    def params(self) -> dict:
        """Derivation params recorded in `derived.parquet` (and matched by selection configs)."""
        return {"K": self.K, "n_sample": self.n_sample, "seed": self.seed}

    def path(self, dataset: str, root: Path | str = "results/features") -> Path:
        """Where the fitted vocabulary is saved (always fitted on the train split)."""
        return Path(root) / dataset / "vocab" / f"{self.backbone}-K{self.K}-n{self.n_sample}-seed{self.seed}.npz"


@dataclass(frozen=True)
class Vocabulary:
    centroids: np.ndarray  # [K, D] float32, unit-norm
    sizes: np.ndarray  # [K] sample patches assigned to each prototype
    objective: float  # mean cosine distance of the sample to its prototype
    meta: dict


def _image_rng(seed: int, image_id: str) -> np.random.Generator:
    """Per-image generator keyed by (seed, image id), so the sample doesn't depend on load order."""
    return np.random.default_rng([seed, zlib.crc32(image_id.encode())])


def flat_patches(feats: dict) -> torch.Tensor:
    """`[Hp, Wp, D]` cached patch grid -> `[Hp*Wp, D]` float32 unit-norm rows (row-major)."""
    p = feats["patch"].float()
    return torch.nn.functional.normalize(p.reshape(-1, p.shape[-1]), dim=-1)


def sample_patches(store: FeatureStore, ids: Sequence[str], n_per_image: int, seed: int) -> np.ndarray:
    """Draw `n_per_image` patches from every image, uniformly without replacement.

    Args:
        store: Raw `FeatureStore` (train split).
        ids: Images to sample from.
        n_per_image: Patches per image (all of them if the grid is smaller).
        seed: Sample seed; each image's draw is seeded by `(seed, crc32(image_id))`.

    Returns:
        `[len(ids) * n_per_image, D]` float32 unit-norm matrix.
    """
    out = []
    for i in tqdm(ids, desc=f"sample {n_per_image} patches/image"):
        p = flat_patches(store.load(i))
        take = _image_rng(seed, i).choice(len(p), size=min(n_per_image, len(p)), replace=False)
        out.append(p[torch.from_numpy(np.sort(take))].numpy())
    return np.concatenate(out)


def _init_plusplus(X: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """Greedy k-means++ seeding on the unit sphere (same variant as
    `selection/prototypicality/scoring.py::_init_plusplus`, in torch)."""
    n = len(X)
    n_trials = 2 + int(np.log(k))
    first = int(torch.randint(n, (1,), generator=gen, device=X.device))
    idx = [first]
    closest = (1.0 - X @ X[first]).clamp_(min=0.0)
    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:
            idx.append(int(torch.randint(n, (1,), generator=gen, device=X.device)))
            continue
        r = torch.rand(n_trials, generator=gen, device=X.device) * total
        cand = torch.searchsorted(torch.cumsum(closest, 0), r).clamp_(max=n - 1)
        cand_dist = torch.minimum(closest[None, :], (1.0 - X[cand] @ X.T).clamp_(min=0.0))  # [trials, N]
        best = int(cand_dist.sum(dim=1).argmin())
        idx.append(int(cand[best]))
        closest = cand_dist[best]
    return X[idx].clone()


def _spherical_kmeans(X: torch.Tensor, k: int, gen: torch.Generator, max_iter: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """One restart of spherical k-means (cosine assignment, normalized-sum update); empty
    prototypes are re-seeded at the points farthest from their own prototype."""
    C = _init_plusplus(X, k, gen)
    labels = None
    for _ in range(max_iter):
        sim, new = (X @ C.T).max(dim=1)
        if labels is not None and torch.equal(new, labels):
            return C, labels, True
        labels = new
        S = torch.zeros_like(C).index_add_(0, labels, X)
        norms = S.norm(dim=1)
        alive = norms > 1e-12
        C[alive] = S[alive] / norms[alive, None]
        dead = torch.nonzero(~alive).flatten()
        if len(dead):
            far = torch.argsort(sim)[: len(dead)]  # lowest cosine to own prototype first
            C[dead] = X[far]
            labels = None
    return C, (X @ C.T).argmax(dim=1), False


def fit_vocabulary(sample: np.ndarray, K: int, seed: int, n_init: int = N_INIT, max_iter: int = MAX_ITER,
                   device: str | None = None) -> Vocabulary:
    """Fit `K` unit-norm prototypes to a patch sample with spherical k-means.

    Runs `n_init` restarts and keeps the lowest mean cosine distance. Float32 throughout
    (the sample is ~1.8 GB; float64 copies would not fit next to a training job).

    Args:
        sample: `[M, D]` unit-norm patches (`sample_patches`).
        K: Number of prototypes.
        seed: Seeds the k-means++ draws of all restarts.
        n_init: Restarts.
        max_iter: Max Lloyd iterations per restart.
        device: Torch device; default CUDA if available.

    Returns:
        The fitted `Vocabulary`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(np.ascontiguousarray(sample, dtype=np.float32)).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    best = None
    n_unconverged = 0
    for r in range(n_init):
        C, labels, converged = _spherical_kmeans(X, K, gen, max_iter)
        n_unconverged += not converged
        objective = float((1.0 - (X * C[labels]).sum(dim=1)).mean())
        log.info("vocab restart %d/%d: mean cosine distance %.5f (converged=%s)", r + 1, n_init, objective, converged)
        if best is None or objective < best[0]:
            best = (objective, C, labels)
    if n_unconverged:
        log.warning("%d/%d vocabulary restarts hit max_iter=%d before converging", n_unconverged, n_init, max_iter)
    objective, C, labels = best
    sizes = torch.bincount(labels, minlength=K).cpu().numpy()
    n_tiny = int((sizes < 0.1 * len(X) / K).sum())
    if n_tiny:
        log.warning("%d/%d prototypes hold < 10%% of the mean cluster size (K may be too large)", n_tiny, K)
    return Vocabulary(centroids=C.cpu().numpy(), sizes=sizes, objective=objective,
                      meta={"n_init": n_init, "max_iter": max_iter, "n_sample_total": len(X),
                            "n_unconverged": n_unconverged})


def save_vocabulary(vocab: Vocabulary, path: Path) -> None:
    """Write centroids, sizes and metadata to `path` (`.npz`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=vocab.centroids, sizes=vocab.sizes,
             meta=json.dumps({**vocab.meta, "objective": vocab.objective}))


def load_vocabulary(path: Path) -> Vocabulary:
    """Read a vocabulary written by `save_vocabulary`."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        return Vocabulary(centroids=z["centroids"], sizes=z["sizes"], objective=meta.pop("objective"), meta=meta)


def load_or_fit_vocabulary(spec: VocabSpec, dataset: str, train_store: FeatureStore, train_ids: Sequence[str],
                           root: Path | str = "results/features") -> Vocabulary:
    """Load the saved vocabulary for `spec`, fitting (and saving) it on the train split if absent.

    Args:
        spec: Vocabulary settings.
        dataset: Dataset name.
        train_store: Raw `FeatureStore` of the train split (the fit always uses train).
        train_ids: Train image ids to sample from.
        root: Features root.

    Returns:
        The `Vocabulary`.
    """
    path = spec.path(dataset, root)
    if path.exists():
        log.info("Loading patch vocabulary %s", path)
        return load_vocabulary(path)
    sample = sample_patches(train_store, sorted(train_ids), spec.n_sample, spec.seed)
    log.info("Fitting patch vocabulary K=%d on %d x %d sampled patches", spec.K, *sample.shape)
    vocab = fit_vocabulary(sample, spec.K, spec.seed)
    vocab = Vocabulary(vocab.centroids, vocab.sizes, vocab.objective,
                       {**vocab.meta, **spec.params(), "backbone": spec.backbone, "n_train_images": len(train_ids)})
    save_vocabulary(vocab, path)
    log.info("Saved patch vocabulary -> %s (cluster sizes min/median/max %d/%d/%d)", path,
             vocab.sizes.min(), int(np.median(vocab.sizes)), vocab.sizes.max())
    return vocab


def assign_patches(feats: dict, centroids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Assign ALL of one image's patches to their nearest prototype.

    Args:
        feats: One image's cached feature dict (needs `patch`).
        centroids: `[K, D]` unit-norm prototypes (torch, any device).

    Returns:
        `(labels [P] int, distance [P] float32 = 1 - max cosine)`, row-major over the grid.
    """
    p = flat_patches(feats).to(centroids.device)
    sim, labels = (p @ centroids.T).max(dim=1)
    return labels.cpu().numpy(), (1.0 - sim).clamp_(0.0, 2.0).cpu().numpy()
