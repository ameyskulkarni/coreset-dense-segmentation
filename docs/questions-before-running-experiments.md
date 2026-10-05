# Answers: pre-experiment questions (2026-09-21)

Answered by reading the actual repo state (plan doc + code), not from general
knowledge. File paths / line refs point at what I read. **No code was changed.**

---

## 1. Hyperparameters — same simple recipe for everything, or tune per model/dataset?

> **Values below are historical (iteration-based).** The recipe now uses
> epochs and a cosine/constant LR schedule instead of poly — see
> `docs/adr/0001-epoch-based-training.md` for current values and
> `configs/recipe/*_proxy.yaml` for the frozen numbers in use. The
> *policy* described here (tune once, freeze, no per-method tuning) is
> unchanged.

You're not wrong — the plan already locks this in as a hard rule, and it's the
right call for this kind of paper.

**§9 of `segmentation-coreset-experiment-plan.md`, two hard rules:**

1. **Tune the training recipe ONCE, on the FULL dataset only.** A small LR sweep
   `{3e-5, 6e-5, 1e-4}` + set the iteration cap via the "discriminativeness check"
   (§4 — see Q2). **Freeze it. Reuse identically for every subset, every method,
   every seed.** Never per-method/per-subset tune the trainer — the plan calls
   this "both unfair and budget-fatal" (tuning per condition would confound
   "data quality" with "recipe fit").
2. **Selection-method hyperparameters** (k-means K, dedup threshold τ, etc.) are
   set by a **cheap label-free proxy** (embedding coverage / silhouette / rare
   pseudo-class recall), never by training a segmenter and picking whatever
   scores highest. Only the final chosen setting gets a real training run.

So: yes, one fixed recipe per **dataset** (not per model/method — model choice
doesn't change the recipe either, per §3: "the model is instrumentation, not
contribution"). The currently-frozen values, already implemented in
`configs/recipe/proxy_20k.yaml`:

| | Cityscapes / ADE20K | CamVid |
|---|---|---|
| optimizer | AdamW | AdamW |
| lr | 6e-5 | 6e-5 |
| lr schedule | poly, power 1.0, 1500-iter linear warmup | same |
| weight decay | 0.01 | 0.01 |
| batch size | 8 | 16 (`proxy_10k_camvid.yaml`) |
| iterations | 20,000 (ADE20K: 40k for one "ceiling" run) | 10,000 |
| precision | AMP fp16 | AMP fp16 |

**Do you need to run experiments to find these?** Only the tiny LR sweep on
100%-data (3 values), plus the one discriminativeness check (Q2) — both one-time,
before Stage 0/1. Nothing else. This is not implemented as a script yet; you'd
run `03_train.py` three times with `--set recipe.lr=...` on full data and eyeball
the loss curve / early mIoU.

---

## 2. Why ~20k iterations, not a fixed epoch count?

> **SUPERSEDED 2026-09-21 — see `docs/adr/0001-epoch-based-training.md`.** After
> discussion, the project switched to fixed-epoch training (matching the closest
> directly-comparable prior work, Nogueira et al. 2505.01225, which fixes epochs
> and reports the resulting compute savings as a result). The reasoning below is
> kept as the historical record of why fixed-iterations was chosen originally —
> read the ADR for the current convention and why it changed.

**§4, verbatim reasoning (as it stood when this doc was written):** "Fix iterations, NOT epochs... Fixing epochs would
give small subsets 10× fewer updates and you'd be measuring schedule length, not
data quality. This is the single most important methodological decision — get it
wrong and the paper is invalid."

The mechanism, concretely: if you fix epochs, a 10% subset sees 10% as many
gradient *steps* as the 100% run (same passes through less data = less total
work). Any mIoU gap you then observe is confounded — you can't tell whether the
subset is worse because the *images it contains* are worse, or simply because it
was trained for less total compute. Fixing iterations forces the loader to
re-visit the smaller subset more times per unit of wall-clock/compute (the
`_infinite_loader` in `segcoreset/train/trainer.py:52-59` does exactly this — it
cycles the same small `DataLoader` indefinitely), so total gradient steps is
identical across every subset size. The *only* thing that varies between runs is
which images are in the subset — which is the actual experimental variable.

Your instinct (100 epochs = same number of dataset passes regardless of size) is
the wrong invariant for this paper: it holds "passes through the data" constant,
not "training compute," and passes-through-data is not what you're trying to
isolate. The plan explicitly names this as the trap to avoid.

**Where the 20k number itself comes from:** not a fixed law, it's set by the
**discriminativeness check** (§4, mandatory day 1, not yet run in this repo):
confirm full-100%@20k clearly beats 10%-random@20k. If they tie, the proxy
schedule is too short to distinguish good from bad subsets → bump to 30-40k and
re-check. So 20k is a starting hypothesis to be validated, not received wisdom —
you should run this check before trusting any Stage 1 result. It's not automated
yet (no `make discriminativeness_check` target — `03_train.py` twice, by hand,
comparing final mIoU).

---

## 3. Is everything on W&B, or only local?

**Both, and there's a real asymmetry to know about:**

- **`03_train.py`** (the normal training path) calls `init_wandb()`
  (`segcoreset/logging/wandb_logger.py`) and logs `train/loss`, `train/lr`,
  `train/imgs_per_sec` every `log_every` (50) steps, `val/*` at each
  `eval_every` (2000) mid-training checkpoint, and `final/*` at the end
  (`segcoreset/train/trainer.py:61-63,97-119`) — **provided W&B is enabled**.
  It **always** also appends one row to `results/runs.csv` regardless of W&B
  status (`scripts/03_train.py:77-98`).
- **`04_eval.py`** (standalone re-eval of an existing checkpoint) does **not**
  touch W&B at all — no import, no `init_wandb` call. It only prints to stdout
  and appends a row to `runs.csv`. If you re-score a checkpoint later, that
  result exists in `runs.csv` but never appears on the W&B dashboard.

- **Default W&B mode is `"online"`** (`wandb_logger.py:20`,
  `configs/experiment/example_ade20k_random20.yaml:14`) — i.e. if you don't
  explicitly set `wandb.mode: offline` or `disabled`, every `03_train.py` run
  will try to sync to the W&B cloud (project `coreset-dense-segmentation`) and
  needs you logged in (`wandb login`) first. Worth deciding now, since running
  50+ experiments will otherwise silently assume cloud logging is wanted and
  configured.

**Bottom line:** `runs.csv` is the durable source of truth for aggregation
(`05_aggregate.py`, not built yet, will read only this file) — treat W&B as a
live-monitoring convenience, not the paper's data source.

---

## 4. SegFormer input size across datasets of different native resolutions

SegFormer-B0's MiT encoder (`segcoreset/models/segformer.py`) is fully
convolutional (overlap patch embeddings, no fixed positional embedding grid
like plain ViT), so it accepts **any input resolution divisible by 32**
(4 stages, each downsampling ×2, plus the initial stride-4 patch embed —
32 = 4·2·2·2). It is not tied to a single canonical input size the way
ImageNet ViT classifiers are.

The repo handles varying dataset resolutions with per-dataset **crop sizes**,
not a shared canonical resize (`segcoreset/data/transforms.py:102-113`,
values in `configs/dataset/*.yaml`):

| Dataset | native size | train crop | divisible by 32? |
|---|---|---|---|
| Cityscapes | 1024×2048 | 512×1024 | yes |
| ADE20K | variable | 512×512 | yes |
| CamVid | 720×960 | 480×480 | yes |

Training pipeline: `RandomScale([0.5,2.0])` → `RandomCrop(crop_size)` →
flip → jitter → normalize (`build_train_transform`). So every batch is a
fixed-size crop *per dataset*, and the crop size is chosen per-dataset in the
plan (§2) — this is intentional, not an oversight: different datasets get
different (fixed) crop sizes, but within one dataset every run uses the same
crop, so it doesn't confound the cross-method comparison.

At eval time there's no fixed size at all: Cityscapes uses sliding-window
tiling at the same 512×1024 crop with 2/3-stride (`sliding_window.py`,
mmseg convention), ADE20K/CamVid do whole-image inference with a configurable
`resize_short_side` (ADE20K: 512) and bilinear-upsample logits back to the
original label size (`evaluator.py:56-64`) — so eval resolution and train crop
size are independent knobs, again per dataset.

Feature-extraction resolution (for DINOv2/CLIP/ResNet/SegFormer-encoder
selection features) is a *separate*, also-fixed-per-extractor size
(e.g. DINOv2 ViT-B/14 uses 518×518, `configs/features/dinov2_vitb14.yaml`) —
unrelated to the training crop size, since selection happens before training
on cached features.

**Nothing to decide here** — the per-dataset crop-size design is already in
place and is the standard way segmentation papers handle this; you'd only
touch it if you added a new dataset.

---

## 5. Are embeddings normalized? Should we only work with normalized ones?

**Yes — every feature extractor L2-normalizes its output before caching it,**
already implemented, no config flag to turn it off:

- DINOv2/DINOv3 (`segcoreset/features/dino.py:73-76`): both `cls` and `patch`
  outputs go through `F.normalize(..., dim=-1)`.
- ResNet-50 (`resnet.py:38`): `F.normalize(feats, dim=-1)`.
- CLIP (`clip_extractor.py:26`): `F.normalize(out.image_embeds, dim=-1)`.
- SegFormer-encoder (`segb0_encoder.py:42`): `F.normalize(feat, dim=-1)` (after
  global-average-pooling the last hidden state).

This matches standard practice for the selection mechanisms the plan uses
(cosine-similarity k-center / farthest-point coverage, SemDeDup's cosine
dedup threshold, prototypicality's distance-to-centroid) — all of these are
cosine-geometry methods, so L2-normalized embeddings are the correct and
already-enforced choice. You don't need to do anything here; just don't
introduce a selector that expects raw (unnormalized) magnitude to carry
signal without checking this assumption first.

---

## 6. What RNG strategy does random selection use — uniform?

**Uniform, via a Fisher–Yates shuffle-then-slice**
(`segcoreset/selection/random_select.py`):

```python
rng = _random.Random(seed)   # separate RNG instance, seeded independently
ids = list(image_ids)
rng.shuffle(ids)
return sorted(ids[:budget])
```

Every image has equal probability of being chosen (no weighting by class
frequency, image size, etc. — this is deliberately the "floor" baseline, §5.1).
`budget = round(ratio * len(image_ids))` (`scripts/02_select.py:45`). The
final `sorted()` doesn't affect *which* images are chosen, only the order they're
stored in — it's there so the cached subset JSON is byte-identical across
re-runs with the same seed (diff-friendly, avoids spurious git changes / cache
misses downstream).

Note the RNG is a **fresh, independently-seeded `random.Random(seed)`**, not the
global `random` module state set by `set_seed()` in `train/trainer.py` — so
selection determinism doesn't depend on training being seeded first, and vice
versa. `image_ids` itself comes from a `sorted(glob(...))` listing per dataset
(e.g. `cityscapes.py:50`), so the input order is stable across machines/runs
too — the shuffle is the only randomness source.

---

## 7. Once I fix a seed, is everything actually reproducible? Any gaps?

**Mostly yes, with one real gap worth knowing about before you rely on it for a paper.**

**What IS covered by `set_seed()`** (`segcoreset/utils/seed.py`, called once
at `Trainer.__init__` with `cfg.recipe.seed_train`):
- `random`, `numpy`, `torch` (CPU + all CUDA devices), `PYTHONHASHSEED`.
- `torch.backends.cudnn.deterministic = True`, `benchmark = False` — trades
  some speed for reproducible convolution algorithms.
- Deliberately **not** using `torch.use_deterministic_algorithms(True)` — the
  seed.py docstring explains this breaks adaptive-pooling backward and
  `interpolate` (both used here: SegFormer's decode-head upsample and the
  eval-time resize). This is a documented, reasoned tradeoff, not an oversight.

**Selection** is independently and separately seeded (`random_select.py`,
Q6) — deterministic given `seed` regardless of training-seed state.

**The gap: DataLoader worker processes — FIXED 2026-09-21.** The recipes set
`num_workers: 8`, and `Trainer._infinite_loader`
(`segcoreset/train/trainer.py`) used to build the `DataLoader` with **no
`worker_init_fn`**; it now passes one (`_seed_worker`) that reseeds each
worker's `random`/`numpy` state from its own torch-assigned seed. Kept below as
the historical record of the gap. On Linux, PyTorch's default multiprocessing
start method is
`fork`, so each worker inherits the *parent's* `random`/`numpy` RNG state at
fork time — meaning the augmentation ops in `transforms.py` (`RandomScale`,
`RandomCrop`, `RandomHorizontalFlip` — all using the plain `random` module, not
a per-worker generator) can produce **correlated or duplicated augmentation
draws across the 8 workers** within a run. This is a known PyTorch gotcha, not
specific to this repo, but it isn't guarded against here. Two consequences:
  1. Two runs with the same seed *should* still reproduce each other
     identically (both forking from the same deterministically-seeded parent
     state, same worker count) — so **seed-to-seed reproducibility likely still
     holds** as long as `num_workers` and hardware/library versions are also
     unchanged.
  2. But *within* a single run, augmentation diversity across the 8 parallel
     workers is weaker than intended (workers 0-7 start from very similar/same
     RNG state), which could very mildly reduce effective augmentation
     variety. This does not threaten cross-method comparisons (it affects
     every run equally), but it's a latent correctness gap if you ever change
     `num_workers` between two runs you intend to compare, or move to a
     platform where the multiprocessing start method is `spawn` instead of
     `fork` (e.g. some macOS/Windows setups) — `spawn` workers do NOT inherit
     parent RNG state and would reseed from OS entropy, silently making runs
     *not* reproducible run-to-run in that case. Since you're on a fixed Linux
     3090 box (`fork` default), this was a minor-severity latent risk rather
     than an active bug — now closed via `worker_init_fn`.

**Two more "seed" params to keep straight (both already logged for provenance):**
- `cfg.recipe.seed_train` — training init/order seed (model init, data
  shuffling, augmentation, AMP internals).
- selection `seed` (CLI `--seed` to `02_select.py`) — controls the subset draw
  itself for stochastic selectors; independent of `seed_train`.
Both are recorded per run: `runs.csv` stores `seed` (train seed) and the subset
filename encodes the selection seed; `config_hash` + `git_commit` are also
captured per run (`utils/config.py`, `utils/git_utils.py`) and `03_train.py`
**refuses to run on a dirty git tree** unless `--allow-dirty`
(`check_clean_tree`, `scripts/03_train.py:49`) specifically so `git_commit` in
`runs.csv` is trustworthy.

**One more determinism-relevant detail:** the periodic mid-training eval subsamples
with `np.random.default_rng(0)` — a **hardcoded** seed 0, independent of
`seed_train` (`evaluator.py:37`, comment: "periodic eval always samples the same
subset"). That's intentional (keeps the mid-training eval subset identical
across all seeds/methods so those numbers are comparable to each other), but it
means mid-training numbers are not affected by your training seed — only the
final full-protocol eval (no subsampling) reflects seed variance.

---

## Not yet built (from README "What's not here yet" + plan gaps found above)

- Selection methods beyond `random`/`full` (prototypicality, k-center,
  SemDeDup, bpp, ZCore, `patch_coverage`) — none exist yet in
  `segcoreset/selection/`.
- `05_aggregate.py` / `06_plots.py` and a `make killshot`/`make lean_core`
  batch launcher.
- The LR sweep (Q1) and discriminativeness check (Q2) — no script automates
  either; both are manual `03_train.py` invocations you'd run and compare by
  hand before trusting any real experiment. Re-run the discriminativeness
  check under the new epoch-based convention — see `docs/adr/0001-epoch-based-training.md`.
