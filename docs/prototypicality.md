# Prototypicality selection — how it works in this repo

This document describes the prototypicality selector **as implemented in this codebase**
(`segcoreset/selection/prototypicality/`): what it computes, how to run it end to end
(embeddings → selection → training → evaluation), how runs and files are named, how to
inspect a selection, and the experiment design (k values, ratios, seeds). Where the
implementation follows Sorscher et al., *"Beyond neural scaling laws: beating power law
scaling via data pruning"* (NeurIPS 2022, §6), that is noted, but the code is the source of
truth here.

All example numbers are real: Cityscapes train (N = 2,975 images), DINOv2 ViT-B/14 CLS
embeddings, k = 20, selection seed 0, keep ratio 0.2 (595 images), unless stated otherwise.

---

## 1. The idea in plain terms

1. Describe every training image with one vector (an embedding from a self-supervised
   model; here DINOv2's CLS token). Similar-looking scenes get similar vectors.
2. Group the vectors into **k clusters**. Each cluster centre is a **prototype** — "the
   typical image of this kind of scene".
3. Score each image by **how far it is from its nearest prototype** (cosine distance).
   - **Small distance → easy / prototypical**: looks like many other images; largely
     redundant with them.
   - **Large distance → hard / atypical**: unusual scene, lighting, layout or content.
4. Keep `budget` images from one end of that ranking — while (by default) guaranteeing every
   cluster a minimum share, so the subset can't collapse onto a few scene types.

**Why this could help.** If a dataset has many near-duplicates (in Cityscapes: consecutive
frames of similar streets), training on all of them wastes compute — a few prototypical
examples carry most of the information. The paper's theory predicts *which end* to keep:

- **Plenty of data left (high keep ratio) → keep the hard ones.** The model sees enough
  typical examples anyway; atypical images add new information.
- **Little data left (low keep ratio) → keep the easy ones.** With only a handful of images
  the model first needs the common cases; a subset of only outliers teaches the exceptions
  without the rule.

Both directions are implemented (`keep: hard` / `keep: easy`) so this can be tested.

**Why it might *not* help for segmentation (the research question).** The score comes from
one *image-level* vector. "Atypical image" is not the same as "image containing rare
classes": an image can be atypical because of lighting or weather while containing only road
and sky, and a small motorcycle in an ordinary street barely moves the CLS vector. The
project's thesis (plan §1) is that image-level label-free selectors like this one don't
reliably beat random for segmentation, and that the shortfall shows up in rare-class mIoU.
Prototypicality is the canonical representative of the "representativeness" family in that
comparison.

---

## 2. Where the code lives

```
configs/selection/
  prototypicality-hard.yaml         keep=hard, balance=0.5   (default preset)
  prototypicality-easy.yaml         keep=easy, balance=0.5
  prototypicality-hard-nobal.yaml   keep=hard, balance=0.0   (paper's unbalanced metric, reference)
  prototypicality-easy-nobal.yaml   keep=easy, balance=0.0

segcoreset/selection/prototypicality/
  __init__.py     public exports
  scoring.py      steps 1-3: normalize -> spherical k-means -> distance to nearest prototype
  pruning.py      step 4: distances -> exactly `budget` kept rows (+ cluster balancing)
  selector.py     PrototypicalityConfig, run() (everything), select() (registry entry), auto_name()
segcoreset/selection/registry.py    SELECTOR_REGISTRY + AUTO_NAMERS + selection_name()

segcoreset/features/embeddings.py   loads the [N, D] embedding matrix; EmbeddingSpec.tag()
segcoreset/features/derived.py      derived.parquet store (written by 00b)
segcoreset/features/store.py        raw per-image .pt cache (written by 00)

scripts/00_extract_features.py       raw DINOv2 cls + patch tokens -> .pt cache
scripts/00b_derive_embeddings.py     .pt cache -> one vector per image -> derived.parquet
scripts/02_select.py                 runs the selector, writes results/subsets/*.json
scripts/02b_inspect_prototypicality.py   same selection, dumps every intermediate; k-sweep
scripts/02c_cluster_diagnostics.py   silhouette / elbow curves of an embedding (diagnostic only)
scripts/03_train.py / 04_eval.py     unchanged; consume the subset file like any other
```

The package is self-contained: removing its two lines in `registry.py` unplugs it. Training
and evaluation code know nothing about prototypicality — they only ever see a list of ids.

---

## 3. The configuration

`configs/selection/prototypicality-hard.yaml` (comments trimmed):

```yaml
name: auto                   # -> proto-hard-k20-bal50-dinov2vitb14-cls (section 6)
method: prototypicality      # registry key -> segcoreset/selection/prototypicality
k: 20                        # number of prototypes (spherical k-means clusters)
keep: hard                   # hard = keep largest distances; easy = keep smallest
balance: 0.5                 # cluster balancing strength in [0, 1] (section 4.5)
n_init: 10                   # spherical k-means restarts; lowest total cosine distance kept
max_iter: 300                # assignment/update iterations per restart
embedding:                   # which saved vectors to cluster (section 5)
  source: derived            # derived.parquet (00b) | raw (.pt cache)
  backbone: dinov2_vitb14
  method: cls                # 00b derivation: cls | patch_mean | kmeans_hist
  params: {}                 # must equal the 00b params exactly
  split: train
```

| field | default if absent | validated |
|---|---|---|
| `k` | 20 | `1 <= k <= N` |
| `keep` | `hard` | `hard` or `easy` |
| `balance` | **0.5** | in `[0, 1]` |
| `n_init` | 10 | — |
| `max_iter` | 300 | — |

`PrototypicalityConfig.from_cfg` (`selector.py`) picks these five fields out of the
`selection:` block. **`balance` defaults to 0.5 everywhere** — in the dataclass and in the two
main presets — so balanced selection is what you get unless you explicitly ask for
`balance: 0` (the `-nobal` presets).

The four presets differ only in `keep` and `balance`. Other settings (k, embedding) are
changed with `--set`, e.g. `--set selection.k=10`; the auto-generated name picks the change
up, so each variant gets its own files (section 6).

**Selection seed.** `--seed` on `02_select.py` seeds the k-means++ draws of all restarts. It
is the *only* source of randomness in the selector; everything after clustering is
deterministic.

---

## 4. How the selection is computed, step by step

`select(image_ids, features, budget, seed, cfg)` → `run(...)` →
`compute_prototype_scores` (scoring.py) → `select_by_score` (pruning.py).

### 4.1 Normalize the embeddings

Input: `X`, an `[N, D]` matrix whose row *i* is `image_ids[i]`'s embedding (2,975 × 768).
It is cast to float64 (float32 sums vary with thread summation order; float64 keeps results
reproducible) and **every row is scaled to length 1** (`l2_normalize`).

*Why normalize if the stored vectors are already normalized?* The scorer accepts **any**
embedding, and not all are unit length: `cls` / `patch_mean` are, but `kmeans_hist`
histograms **sum** to 1 (an even 256-bin histogram has length 0.0625), and future embeddings
are unknown. Normalizing inside guarantees cosine geometry for every input.

*Is normalizing twice harmful?* No — dividing a length-1 vector by its length divides by 1:

```
v = (0.6, 0.8)   length √(0.36 + 0.64) = 1   →   v / 1 = (0.6, 0.8)   unchanged
```

Measured: stored CLS vectors differ from length 1 by at most 1.1 × 10⁻⁷ (float32 rounding);
re-normalizing changes any value by at most 1.5 × 10⁻⁸. The only failure case is an all-zero
vector (no direction), which raises a clear error.

### 4.2 Spherical k-means (cosine k-means)

`scoring.py::spherical_kmeans`, run `n_init` times:

1. **Seeding — greedy k-means++** (`_init_plusplus`, the variant sklearn uses): the first
   centre is a random image; each next centre is chosen from `2 + log(k)` candidates drawn with
   probability proportional to their cosine distance to the nearest centre so far, keeping
   the candidate that most reduces the total. (For unit vectors ‖x − c‖² = 2(1 − cos), so this
   is exactly standard k-means++.)
2. **Assign** every image to the prototype with the **highest cosine similarity**.
3. **Update** every prototype to the **mean of its members, rescaled to length 1**.
4. Repeat 2–3 until no assignment changes (a fixed point) or `max_iter` is reached.
5. **Empty prototype** (no members): re-seeded at the image farthest from its own prototype.
6. Of the `n_init` restarts, keep the one with the **lowest total cosine distance**
   (ties: the earliest).

*Why rescale the prototype in step 3?* The mean of unit vectors is shorter than 1:

```
a = (1, 0)   b = (0, 1)              both length 1, 90° apart
mean = (0.5, 0.5)                    length √0.5 = 0.707
```

The score is cosine distance, computed as a dot product, and a dot product equals the cosine
only when both vectors have length 1:

```
a · mean       = 0.5     → 1 − 0.5   = 0.500   wrong
a · unit(mean) = 0.707   → 1 − 0.707 = 0.293   correct cosine distance
```

On Cityscapes the cluster means have length 0.56–0.81 (median 0.74): tight clusters have
longer means, loose clusters shorter. Without rescaling, images in loose clusters would look
artificially "hard" — in a test with the old implementation this changed 11% of a subset.

*Why spherical k-means instead of standard k-means?* Standard (Euclidean) k-means assigns
points by distance to the *unscaled* mean, which depends on the mean's length as well as the
angle (‖x − c‖² = 1 + ‖c‖² − 2‖c‖cos). The score is pure cosine, so with standard k-means a
few borderline images ended up assigned to one cluster but nearest (by cosine) to another.
Spherical k-means assigns and updates by cosine, so **clustering and scoring use one
geometry**: every image's cluster *is* its nearest prototype, and the objective being
minimized (total cosine distance) is exactly what the score measures. The paper's recipe —
cluster self-supervised embeddings, score by cosine distance to the nearest centroid — is
unchanged.

Comparison with the previous sklearn (Euclidean k-means) implementation, k = 20, seed 0,
keep-hard, balance 0.5 — share of selected images that are the same:

| keep ratio | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|---|
| spherical vs sklearn | 95% | 93% | 92% | 89% | 83% |
| *context: sklearn seed 0 vs seed 1* | 96% | 94% | 91% | 89% | 83% |

Switching algorithms changes the subset about as much as changing the k-means seed — within
the method's own randomness. Clustering quality is equal: total cosine distance averaged over
seeds 0–2 is lower for spherical at k = 10 and 50 and within 0.05% at k = 20.

### 4.3 Score

`distance[i] = 1 − cos(image i, its prototype)`, clipped to `[0, 2]`. 0 = same direction as its
prototype; larger = more atypical.

Output: `PrototypeScores` with `distance [N]`, `cluster [N]`, `centroids [k, D]` (unit
length) and `inertia` = Σ‖x − c‖² over images and their unit prototypes = 2 × Σ distance.

Warnings you may see: `N/10 spherical k-means restarts hit max_iter=...` (didn't converge;
raise `max_iter`), `N/k prototypes have no members` (k too large for the data, or many
duplicate images).

Real distribution (k = 20): distance min/median/max = 0.080 / 0.234 / 0.863; cluster sizes
min/median/max = 51 / 150 / 271.

### 4.4 Rank

All N images are sorted from "keep first" to "prune first" (`keep_priority`):
- `keep: hard` → descending distance (most atypical first)
- `keep: easy` → ascending distance (most prototypical first)

Ties are broken by row index (the dataset's sorted image-id order), so the order is fully
deterministic. With `balance: 0` the first `budget` images of this order are kept — a plain
global top-`budget`. The result is returned as sorted image ids, and an assert checks that
exactly `budget` images were selected.

### 4.5 Cluster balancing (`balance`, default 0.5)

**What it guarantees.** Each cluster first receives a **quota** of
`balance × its fair share`, where its **fair share** is what it would keep under uniform
pruning (its size × the keep ratio):

```
quota_c = min(n_c, floor(balance * n_c * budget / N))
```

Then:
1. Each quota is filled with that cluster's own best-ranked members (same hard/easy order).
2. The remaining `budget − Σ quota` slots go to the best-ranked images globally, skipping
   those already chosen.

`balance` is **not** "keep 50% of each cluster's images" — that would be impossible when
pruning more than 50% (if every cluster kept half, the total would be at least half). Because
the quota scales with the keep ratio, it works at every ratio. Example: the most typical
cluster (cluster 13, 177 images):

| keep ratio | budget | fair share | quota (balance 0.5) | as % of the cluster |
|---|---|---|---|---|
| 0.7 | 2,082 | 123.9 | 61 | 34% |
| 0.5 | 1,488 | 88.5 | 44 | 25% |
| 0.3 | 892 | 53.1 | 26 | 15% |
| 0.2 | 595 | 35.4 | 17 | 10% |
| 0.1 | 298 | 17.7 | 8 | 5% |

The quota is a **floor, not a cap**: a cluster can get far more through the global fill.
Since each quota is at most `balance` × the fair share, quotas sum to at most
`balance × budget` (288 of 595 here, a bit under half because of rounding down), so the
budget can always be met. `balance = 0` → no quotas; `balance = 1` → every cluster pruned by
the same fraction, the ranking only decides which images within each cluster.

**What it prevents** — real selections at 20%:

| config | cluster 0 (81 imgs, most atypical) | cluster 18 (67 imgs) | cluster 13 (177 imgs, most typical) | clusters with 0 kept |
|---|---|---|---|---|
| `prototypicality-hard-nobal` | 73 (90%) | 50 | 12 (7%) | 0 |
| `prototypicality-easy-nobal` | 0 | 0 | 88 | **3** |
| `prototypicality-hard` (balance 0.5) | 72 | 50 | 17 (= its quota) | 0 |
| `prototypicality-easy` (balance 0.5) | 8 | 6 | 83 | 0 |

Unbalanced keep-easy drops three scene types entirely; unbalanced keep-hard nearly empties
the typical clusters. Balancing guarantees every cluster a share in both directions.

**Why 0.5, and how to describe it.** It is the label-free analogue of the paper's class
balancing (Sorscher et al., App. H — check the exact definition there before citing), with
k-means clusters in place of classes. It makes the baseline *stronger*: if balanced
prototypicality still fails to beat random, the negative result can't be blamed on subset
collapse. Describe it as *"prevents the subset from collapsing onto a few scene types"* —
**not** as helping rare classes: clusters are CLS scene types, not semantic classes, so
balancing guarantees scene coverage, not motorcycles. The value is fixed by this label-free
reasoning before any run; it must not be chosen by looking at rare-class pixel counts (that
would leak labels into a label-free method's settings).

**Known limitation.** The rounding down gives tiny clusters no guarantee at low ratios: at
k = 50 the smallest cluster has ~8–13 images, so at keep ratio 0.1 its quota is
floor(0.5 × 10 × 0.1) = 0. The inspect tool's `clusters_with_zero_kept` shows when it happens.

---

## 5. Where the embeddings come from

The selector never reads files. `02_select.py` reads the config's `embedding:` block,
`segcoreset/features/embeddings.py::load_embedding_matrix` loads the matrix, and the selector
receives a plain `[N, D]` float32 array whose row *i* is `image_ids[i]`.

**Raw cache** (`00_extract_features.py`) — `results/features/<dataset>/<backbone>/<split>/`:
one `<image_id>.pt` per image holding `{"cls": [768], "patch": [37, 37, 768]}`, both
L2-normalized float32 (~4.2 MB per image, ~12 GB for Cityscapes train; the patch grid is
>99.9% of that), plus `index.json` (all ids, dataset, representation, split, id → image path).
Images are resized (bicubic, aspect ratio not preserved) to 518×518 and ImageNet-normalized.

**Derived table** (`00b_derive_embeddings.py`) — `results/features/<dataset>/derived.parquet`:
one row per (image, derivation) with columns `image_id, image_path, dataset, split, backbone,
source, method, params, dim, embedding (list<float32>), config_hash, created_at`.

| `method` | vector per image | dim |
|---|---|---|
| `cls` | the cached CLS vector, as is | 768 |
| `patch_mean` | mean of the 37×37 patch grid, re-L2-normalized | 768 |
| `kmeans_hist` | MiniBatchKMeans over all train patches → K pseudo-classes; normalized histogram of the image's patches (`params: {K, seed}`) | K |

Re-running a derivation replaces its rows. Cityscapes `cls` is ~9.8 MB (2,975 × 768 float32
plus metadata).

**How `embedding:` is resolved.** `source: derived` filters the table on `split`,
`backbone`, `method` and `params` (compared as canonical sorted-key JSON, so `{}` ≠
`{K: 256, seed: 0}`) and orders rows to match `image_ids`. `source: raw` reads tensor `key`
(default `cls`) from each `.pt`; only 1-D tensors are allowed, so `patch` must go through
`00b` first. Every candidate image must have exactly one embedding (missing or duplicate ids
raise), and NaN/Inf values raise — selection never silently skips images.

---

## 6. Naming: selections, subsets and runs

Names are built so that **every file and run can be identified from its name alone**, and
so that two different selections can never share a file.

### 6.1 Selection name (`name: auto`)

All prototypicality presets use `name: auto`. `selection_name()` (`registry.py`) calls
`prototypicality.auto_name()`, which encodes every setting that changes the subset:

```
proto-<keep>-k<k>-bal<balance×100>-<embedding tag>[-ninit<n>][-maxiter<n>]
```

| part | meaning | examples |
|---|---|---|
| `proto` | prototypicality | |
| `<keep>` | direction | `hard`, `easy` |
| `k<k>` | number of prototypes | `k10`, `k20`, `k50` |
| `bal<…>` | balance × 100 | `bal50` (0.5), `bal0` (none), `bal25` |
| embedding tag | `EmbeddingSpec.tag()`: backbone + derivation + params, underscores removed | `dinov2vitb14-cls`, `dinov2vitb14-patchmean`, `dinov2vitb14-kmeanshist-K256-seed0`, `raw-dinov2vitb14-cls`; a non-train split appends `-val` |
| `-ninit` / `-maxiter` | only when different from the defaults (10 / 300) | `-ninit20` |

Examples (all verified):

| command | selection name |
|---|---|
| `--selection prototypicality-hard` | `proto-hard-k20-bal50-dinov2vitb14-cls` |
| `--selection prototypicality-hard --set selection.k=10` | `proto-hard-k10-bal50-dinov2vitb14-cls` |
| `--selection prototypicality-easy` | `proto-easy-k20-bal50-dinov2vitb14-cls` |
| `--selection prototypicality-hard-nobal` | `proto-hard-k20-bal0-dinov2vitb14-cls` |
| `... --set selection.embedding.method=patch_mean` | `proto-hard-k20-bal50-dinov2vitb14-patchmean` |
| `... --set selection.embedding.method=kmeans_hist selection.embedding.params.K=256 selection.embedding.params.seed=0` | `proto-hard-k20-bal50-dinov2vitb14-kmeanshist-K256-seed0` |

Because the name is generated, `--set` overrides never collide with another variant's files.
You can still set an explicit name (`--set selection.name=my-name`); it must not contain `_`.
`random` and `full` keep their fixed names.

### 6.2 Where the name appears

| artifact | pattern | example |
|---|---|---|
| subset file | `results/subsets/<dataset>_<selection>_<ratio>_<sel-seed>.json` | `cityscapes_proto-hard-k20-bal50-dinov2vitb14-cls_0.2_0.json` |
| `runs.csv` `selection_method` | the selection name (parsed from the subset filename) | `proto-hard-k20-bal50-dinov2vitb14-cls` |
| inspect outputs | `results/selection_debug/<dataset>_<selection>_s<seed>_{scores,clusters}.csv`, `..._k_sweep.csv` | |
| 02c outputs | `results/selection_debug/<dataset>_<embedding tag>_cluster_diagnostics.{csv,png}` | `cityscapes_dinov2vitb14-cls_cluster_diagnostics.png` |

`03_train.py` splits the subset filename on `_` to recover the selection name and ratio —
which is why selection names never contain `_`.

### 6.3 Run name (`--run-name`)

Use **`<ds>_<selection>_r<ratio>_s<seed>`**, matching the existing random-baseline runs
(`cs_random_r0.2_s0`):

```
cs_proto-hard-k20-bal50-dinov2vitb14-cls_r0.2_s0
└┬┘└────────────────┬────────────────┘└┬─┘└┬┘
dataset         selection name       ratio seed
```

- `cs` = Cityscapes (`ade` for ADE20K, `cv` for CamVid).
- One seed number for both selection (`02_select --seed`) and training (`03_train --seed`).
  If you ever decouple them, write both: `..._r0.2_sel1_s0`.
- Another architecture (Stage 4): insert it after the dataset, e.g.
  `cs_deeplabv3plus-r50_proto-hard-k20-bal50-dinov2vitb14-cls_r0.2_s0`.

The run name becomes the checkpoint directory (`results/checkpoints/<run-name>/`), the W&B
run name and `runs.csv`'s `run_id`. Without `--run-name`, `03_train.py` uses
`<dataset>_<model>_<selection>_r<ratio>_s<seed>_<config-hash>`, which is also unambiguous
but longer.

---

## 7. Running it end to end (Cityscapes)

Activate the environment first: `conda activate coreset-dense-segmentation`.

### Step 0 — prerequisites (once per dataset)

```bash
python scripts/00_extract_features.py --dataset cityscapes --features dinov2_vitb14 --split train  # GPU, ~12 GB, resumable
python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 --method cls  # CPU, seconds
python scripts/01_compute_rare_classes.py --dataset cityscapes                                     # rare-class list for eval
```

On this machine all three outputs already exist.

### Step 1 — inspect (optional, recommended, ~5 s)

```bash
python scripts/02b_inspect_prototypicality.py --dataset cityscapes --selection prototypicality-hard --ratio 0.2
```

See section 8.

### Step 2 — select

```bash
python scripts/02_select.py --dataset cityscapes --selection prototypicality-hard --ratio 0.2 --seed 0
# -> results/subsets/cityscapes_proto-hard-k20-bal50-dinov2vitb14-cls_0.2_0.json
```

What happens (`scripts/02_select.py`):
1. Builds the config (`dataset` + `selection` presets + any `--set` overrides).
2. Lists the full train split's ids (sorted by path) → N = 2,975.
3. `budget = round(ratio × N)`.
4. Resolves the selection name (`name: auto` → section 6.1; rejects `_`).
5. Loads the embedding matrix from the `embedding:` block.
6. Calls `prototypicality.select(...)` (~1.7 s) and writes the subset file:

   ```json
   {
     "image_ids": ["...", "..."],                    // 595 ids, sorted
     "dataset": "cityscapes", "method": "prototypicality",
     "ratio": 0.2, "seed": 0, "n_images": 595, "n_total": 2975,
     "name": "proto-hard-k20-bal50-dinov2vitb14-cls",
     "selection_cfg": {"k": 20, "keep": "hard", "balance": 0.5, "n_init": 10, "max_iter": 300,
                       "embedding": {...}, ...},
     "selection_seconds": 1.65
   }
   ```

**Overwrite protection:** an existing file with identical ids is kept ("Subset unchanged");
one with different ids is only overwritten with `--force`. Re-running the same command is
therefore safe and confirms determinism.

Budgets on Cityscapes (Python's `round` rounds halves to even):

| keep ratio | pruning rate | images | steps/epoch (bs 8) | total steps (100 epochs) |
|---|---|---|---|---|
| 1.0 | 0% | 2,975 | 371 | 37,100 |
| 0.7 | 30% | 2,082 | 260 | 26,000 |
| 0.5 | 50% | 1,488 | 186 | 18,600 |
| 0.3 | 70% | 892 | 111 | 11,100 |
| 0.2 | 80% | 595 | 74 | 7,400 |
| 0.1 | 90% | 298 | 37 | 3,700 |

`ratio` in this code is always the fraction **kept**; papers usually quote the pruning rate
(fraction removed). Ratio 1.0 is the full dataset for every method — reuse the existing
full-data runs as the shared reference instead of re-running them.

### Step 3 — commit the subset, then train

`03_train.py` refuses to run on a dirty git tree (untracked files count), because the
`git_commit` column of `runs.csv` must identify the code and inputs a run used. Subset files
are real inputs and are committed:

```bash
git add results/subsets/cityscapes_proto-hard-k20-bal50-dinov2vitb14-cls_0.2_0.json
git commit -m "Add cityscapes proto-hard-k20-bal50-dinov2vitb14-cls 0.2 subset, seed 0"

python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
    --subset results/subsets/cityscapes_proto-hard-k20-bal50-dinov2vitb14-cls_0.2_0.json \
    --seed 0 --run-name cs_proto-hard-k20-bal50-dinov2vitb14-cls_r0.2_s0
```

What happens:
- Selection name and ratio are parsed from the subset filename and logged to W&B and
  `runs.csv` (`selection_method`, `ratio`).
- `--seed` is the **training** seed (`recipe.seed_train`: init, data order, augmentation),
  independent of the selection seed in the filename; use the same number for both.
- The training set is restricted to the subset ids (an id not in the split raises).
- **Epoch-based schedule** (ADR 0001): 100 epochs over the *subset*, 6 warmup epochs, poly LR
  decay, lr 1e-4, batch size 8 (`configs/recipe/cityscapes_proxy.yaml`). A 20% subset does
  ~20% of the full run's gradient steps (table above).
- Every 5 epochs: a cheap monitoring eval (whole-image, short side 512, fixed 100 val images)
  — a trend line only, **not comparable** to the final Cityscapes number.
- At the end: `final.pt`, then the **full eval protocol** — all 500 val images, sliding window
  at 512×1024 with stride 341×683, plus boundary F-score.
- One row is appended to `results/metrics/runs.csv` (gitignored) with `git_commit`,
  `config_hash`, W&B id/URL, `selection_method`, `ratio`, `seed`, `n_images`, `epochs`,
  `iterations`, `lr`, `miou`, `rare_class_miou`, `pixel_acc`, `boundary_f`,
  `per_class_iou_json`, `gpu_hours_train`.
- Checkpoints and the resolved `config.yaml` go to `results/checkpoints/<run-name>/`.

### Step 4 — evaluation

The final evaluation runs inside `03_train.py`. `04_eval.py` only re-scores an existing
checkpoint and appends a **new** row (`run_id` = `<run-name>_eval`):

```bash
python scripts/04_eval.py --run-dir results/checkpoints/cs_proto-hard-k20-bal50-dinov2vitb14-cls_r0.2_s0
```

Metrics (`segcoreset/eval/`):
- **mIoU** over the 19 classes, from a confusion matrix over all val pixels (ignore 255).
- **rare_class_miou** (the thesis metric): mean IoU over the frozen bottom-5 classes by train
  pixel frequency — motorcycle, rider, traffic light, train, bus
  (`results/rare_classes/cityscapes.json`); omitted silently if that file is missing.
- **pixel_acc**, **boundary_f**, per-class IoU.

A selector "wins" only if its mean across seeds lies outside random's ±1 std band at the
same ratio (plan §7). The random band for Cityscapes is in `runs.csv` (`cs_random_*`).

### Step 5 — the batch script

Following the repo convention (`scripts/experiments/*.sh`, gitignored), for the grid in
section 10. Subsets are generated and committed first (fast, CPU), so the tree is clean for
the whole training batch:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

RATIOS=(0.7 0.5 0.3 0.2 0.1)
SEEDS=(0 1 2)
EMB=dinov2vitb14-cls
# "<preset> <k>" pairs: main grid (keep-hard, 3 k) + direction check (keep-easy, k=20)
CELLS=("prototypicality-hard 10" "prototypicality-hard 20" "prototypicality-hard 50"
       "prototypicality-easy 20")
# optional link to the paper's unbalanced metric: CELLS+=("prototypicality-hard-nobal 20")

name_of() {  # preset k -> selection name (must match auto_name; checked below)
  local keep bal
  keep=$(sed -E 's/prototypicality-(hard|easy).*/\1/' <<<"$1")
  [[ "$1" == *-nobal ]] && bal=0 || bal=50
  echo "proto-${keep}-k${2}-bal${bal}-${EMB}"
}

echo "=== 1/2 select + commit subsets ==="
for cell in "${CELLS[@]}"; do read -r preset k <<<"$cell"
  for ratio in "${RATIOS[@]}"; do for seed in "${SEEDS[@]}"; do
    python scripts/02_select.py --dataset cityscapes --selection "$preset" --set selection.k="$k" \
        --ratio "$ratio" --seed "$seed"
    subset="results/subsets/cityscapes_$(name_of "$preset" "$k")_${ratio}_${seed}.json"
    [[ -f "$subset" ]] || { echo "expected $subset — name_of() out of sync with auto_name()"; exit 1; }
    git add "$subset"
  done; done
done
git diff --cached --quiet || git commit -q -m "Add cityscapes prototypicality subsets (spherical k-means, ${EMB})"

echo "=== 2/2 train ==="
for cell in "${CELLS[@]}"; do read -r preset k <<<"$cell"; name=$(name_of "$preset" "$k")
  for ratio in "${RATIOS[@]}"; do for seed in "${SEEDS[@]}"; do
    python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
        --seed "$seed" --subset "results/subsets/cityscapes_${name}_${ratio}_${seed}.json" \
        --run-name "cs_${name}_r${ratio}_s${seed}"
  done; done
done
```

---

## 8. Inspecting a selection: `scripts/02b_inspect_prototypicality.py`

### What it is and why to run it

It runs **exactly the same code path** as `02_select.py` (same config composition, embedding
loader, `run()` function and name resolution), but instead of writing a subset it dumps
everything `select()` normally throws away. It doesn't train, writes nothing under
`results/subsets/`, and its output directory `results/selection_debug/` is gitignored, so it
never dirties the tree. It runs on CPU in seconds.

Run it to (1) **sanity-check a selection before spending GPU-hours** — does the ranking look
sensible, is the subset collapsing onto a few clusters or dropping scene types? — and (2) to
produce the **k-sweep** robustness table (section 9) without training.

### Arguments

| flag | default | meaning |
|---|---|---|
| `--dataset` | required | dataset config |
| `--selection` | `prototypicality-hard` | any prototypicality preset |
| `--ratio` | none | if given, also mark which images would be selected |
| `--seed` | 0 | k-means seed |
| `--top` | 10 | how many most/least prototypical ids to print |
| `--k-sweep` | none | k values for the robustness proxy, e.g. `5 10 20 50 100` |
| `--sweep-seeds` | `0 1 2` | k-means seeds used in the sweep |
| `--features-root` | `results/features` | where embeddings live |
| `--out-dir` | `results/selection_debug` | output directory |
| `--set` | — | config overrides, e.g. `selection.k=50` |

### Outputs

**`<dataset>_<selection>_s<seed>_scores.csv`** — one row per image, sorted by keep-priority:
`image_id`, `cluster` (its prototype), `distance` (1 − cos), `keep_priority_rank` (0 = kept
first), `selected` (only with `--ratio`).

**`<dataset>_<selection>_s<seed>_clusters.csv`** — one row per prototype: `size`,
`dist_mean`, `dist_min`, `dist_max`, `n_selected` (with `--ratio`), `quota`.

**stdout**: the selection name; a summary (config and seed, `n_total`, `n_selected`, cluster
size min/median/max, distance min/median/max, `selected_distance_min/max`,
`n_from_cluster_quota`, `clusters_with_zero_kept`, `kmeans_inertia`); the per-cluster table;
the `--top` most and least prototypical ids.

Real example (`prototypicality-hard`, ratio 0.2):

```
Selection: proto-hard-k20-bal50-dinov2vitb14-cls
Summary: {'k': 20, 'keep': 'hard', 'balance': 0.5, ..., 'n_selected': 595,
          'cluster_size_min/median/max': [51, 150.0, 271],
          'distance_min/median/max': [0.0796, 0.2336, 0.8634],
          'selected_distance_min/max': [0.2812, 0.8634],
          'n_from_cluster_quota': 288, 'clusters_with_zero_kept': 0, 'kmeans_inertia': 1523.0416}

3 MOST prototypical (easy):            3 LEAST prototypical (hard):
  bremen_000193_000019      0.0796       tubingen_000006_000019   0.7904
  dusseldorf_000123_000019  0.0888       darmstadt_000066_000019  0.8051
  ulm_000049_000019         0.0914       aachen_000152_000019     0.8634
```

### What to look at

- **Open the most and least prototypical images**
  (`/home/cognition/datasets/cityscapes/leftImg8bit/train/<city>/<id>_leftImg8bit.png`).
  Easy ones should look like ordinary streets; hard ones visibly unusual. If not, the
  embedding isn't capturing what you think.
- **`n_selected` vs `size` per cluster** — concentration in a few clusters, or missing ones.
- **`clusters_with_zero_kept`** — non-zero means whole scene types are absent (expected with
  `-nobal` keep-easy; with balance 0.5 only for tiny clusters at high k / low ratio).
- **`selected_distance_min/max`** — the slice of the distance distribution the subset covers.

### Cluster structure diagnostics: `scripts/02c_cluster_diagnostics.py`

```bash
python scripts/02c_cluster_diagnostics.py --dataset cityscapes          # ~2 min
python scripts/02c_cluster_diagnostics.py --dataset cityscapes --set selection.embedding.method=patch_mean
```

Silhouette (cosine) and inertia per k, averaged over seeds, using exactly the selector's
clustering; writes a CSV and a two-panel plot and prints the silhouette argmax and a heuristic
elbow. It is a **diagnostic of how much cluster structure an embedding has** — not used to
choose k (section 9). On DINOv2-B CLS, silhouette peaks at k = 3 with only 0.12 (below ~0.25 =
no substantial structure) and inertia falls smoothly with no real elbow: Cityscapes scenes
form one continuous cloud. An improved embedding showing a clear silhouette peak would itself
be evidence that it adds structure.

---

## 9. Choosing k

### Decision

**Every embedding method is run at the same three fixed values, k ∈ {10, 20, 50}** (ADE20K:
decide its values the same way before running, e.g. {50, 100, 250}). k is **not** tuned per
embedding.

Reasons:
- **Nothing is tuned, so nothing can be challenged.** The values are log-spaced (a 5× range)
  inside the region where k-means is stable on this data (below), chosen before any training.
- **No dependence on the number of classes.** The classification rule of thumb "k ≈ number of
  classes" doesn't transfer: every Cityscapes image contains most of the 19 classes, and CLS
  clusters are scene types, not classes. It is not used as a justification.
- **Silhouette / elbow are not used to choose k**: on this data they show no clear optimum
  (section 8), they measure cluster separation rather than selection quality, and a different
  k per embedding would confound "better embedding" with "different k".
- **Sensitivity to k becomes a reported result**: if an embedding wins at all three k, the
  "was it just k?" question is answered; if embeddings prefer different k, the table shows it.

Reporting rules:
- Report **every k separately** (each against random's ±1 std band) or the **mean over the
  three k** as the method's headline number.
- **Never "best of three k" against random**: the maximum of three noisy results beats a
  single random result more often than chance, and picking k on the val split (the reported
  eval) is test-set tuning. A best-k column is fine as extra information only.

Excluded values, with measured reasons: k = 5 is too coarse (5 prototypes for street scenes)
and diverges most from the rest; k = 100 is unstable (clusters of 2 images; the seed alone
swaps ~25–30% of the subset).

### The k-sweep (label-free robustness check, no training)

```bash
python scripts/02b_inspect_prototypicality.py --dataset cityscapes --selection prototypicality-hard \
    --ratio 0.2 --k-sweep 5 10 20 50 100 --sweep-seeds 0 1 2
```

It answers: **"if I had picked a different k, or a different k-means seed, would I train on
different images?"** It runs the selection for every (k, seed) pair and writes
`<dataset>_<selection>_k_sweep.csv`, one row per k:

| column | question |
|---|---|
| `min_cluster_size`, `median_cluster_size` | is k so large that clusters are tiny or empty? |
| `spearman_across_seeds` | at this k, is the typical→atypical **ranking** of all images stable across k-means seeds? (1 = same order) |
| `jaccard_across_seeds` | at this k, do different seeds select the same **subset**? (1 = identical) |
| `spearman_vs_k<ref>` / `jaccard_vs_k<ref>` | does this k rank / select like the reference k (the preset's k = 20)? |

**Jaccard** = shared images ÷ images in either subset. For equal-size subsets a Jaccard of J
means a fraction 2J / (1 + J) of the images are shared (J = 0.73 → 84%). **Compare it to
chance, not to 1.0**: two random subsets at keep ratio r have expected Jaccard r / (2 − r) —
0.11 at 0.2 (≈ 20% shared), 0.33 at 0.5, 0.82 at 0.9 — so run the sweep at a low ratio. Spearman
uses distances and is the same for hard and easy; Jaccard depends on direction.

**Result** (spherical k-means, balance 0.5, keep 20%, seeds 0/1/2), as **% of images shared**
(chance ≈ 20%):

| k | smallest cluster | hard: seed vs seed | hard: vs k=20 | easy: seed vs seed | easy: vs k=20 | Spearman vs k=20 |
|---|---|---|---|---|---|---|
| 5 | 320 | 100% | 76% | 99% | 65% | 0.87 |
| 10 | 120 | 90% | 81% | 86% | 74% | 0.93 |
| 20 | 43 | 89% | — | 81% | — | — |
| 50 | 8 | 79% | 76% | 74% | 69% | 0.89 |
| 100 | 2 | 76% | 69% | 70% | 61% | 0.82 |

- Stability always rises as k shrinks (k = 1 would be 100% stable and useless), so it is a
  **minimum requirement, not the objective** — it rules out unstable k, it doesn't pick k.
- k = 10–50 select largely the same images as k = 20 (69–81% shared vs ~20% by chance), so the
  three chosen values are distinct settings of one method, not different methods.
- Keep-easy is less seed-stable than keep-hard (81% vs 89% at k = 20): its subset sits in the
  dense centre of the clusters, where small prototype shifts reorder many images.

Re-run the sweep for each new embedding (`--set selection.embedding.method=...`) and report
the table per embedding.

---

## 10. Experiment plan

- **Keep ratios** {0.7, 0.5, 0.3, 0.2, 0.1} = pruning rates {30, 50, 70, 80, 90}% — the
  standard grid of the label-free pruning literature (CCS and work comparing against it;
  verify each baseline paper's exact grid). Full data (1.0) is the shared reference line.
- **Seeds** 0, 1, 2 (same number for selection and training).

| role | preset | k | runs | GPU-hours |
|---|---|---|---|---|
| main grid | `prototypicality-hard` (balance 0.5) | 10, 20, 50 | 45 | ~34–43 |
| direction check | `prototypicality-easy` (balance 0.5) | 20 | 15 | ~11–14 |
| link to the paper's unbalanced metric (optional) | `prototypicality-hard-nobal` | 20 | 15 | ~11–14 |

Per-run cost uses measured training times (`runs.csv`: 0.91 / 0.68 / 0.44 / 0.33 / 0.21 h at
0.7 / 0.5 / 0.3 / 0.2 / 0.1) plus ~0.25–0.45 h of final eval → ~4.1 h per seed across the five
ratios, ~12–14 h per (preset, k) with 3 seeds.

**Why keep-easy at all.** Most of these ratios are in the small-data regime where the
paper's theory says keep-easy should win. Without it, a reviewer can say the baseline ran in
the wrong direction. If keep-easy also fails to beat random, the claim gets stronger; if it
beats keep-hard at small ratios, extend it to k = 10 and 50 there. Run it early.

When testing a new embedding method: change only `selection.embedding`, re-run the k-sweep,
and run the same grid; the auto-generated names keep every variant separate.

---

## 11. Provenance: what is recorded where

| what | where |
|---|---|
| full selection config (k, keep, balance, n_init, max_iter, embedding block), selection seed, timing | the subset JSON (`selection_cfg`, `seed`, `selection_seconds`) — committed |
| everything that changes the subset, human-readable | the selection name, in the subset filename, `runs.csv` `selection_method` and the run name |
| training config, training seed | `results/checkpoints/<run>/config.yaml`, W&B config, `runs.csv` `config_hash` / `seed` |
| code version | `runs.csv` `git_commit` (trustworthy because of the dirty-tree gate) |

Known gaps:
- `runs.csv`'s `representation` column stays empty for prototypicality runs (training doesn't
  load a `features` config); the embedding is identifiable through the selection name.
- `runs.csv`'s `selection_seconds` is not filled by `03_train.py`; the value is in the subset
  JSON.
- `results/features/` (including `derived.parquet`) is gitignored; the committed subset JSONs
  are the inputs, and embeddings are regenerable with `00`/`00b`.

---

## 12. Common errors and warnings

| message | cause | fix |
|---|---|---|
| `No derived embeddings at results/features/<ds>/derived.parquet` | `00b` not run | run `00b_derive_embeddings.py` |
| `No rows in ... match derived[...]` (lists available derivations) | method / params / backbone / split mismatch | run `00b` with those settings or fix the `embedding:` block |
| `N/M image ids have no ... embedding` | raw cache incomplete | re-run `00_extract_features.py` (it resumes) |
| `prototypicality needs an [N, D] embedding matrix` / `...an embedding: block to build its name` | no `embedding:` block | add one |
| `selection.name '...' must not contain '_'` | underscore in an explicit name | use `-` |
| `selection.name: auto is not supported for method ...` | `auto` on random/full | set a name |
| `Refusing to overwrite ...` | same name, different ids (code changed since the subset was made, or an explicit name reused) | check why; `--force` only if intended |
| `k=... must be in [1, N=...]` | k larger than the candidate pool | lower k |
| `RuntimeError` about a dirty tree in `03_train.py` | uncommitted code, configs or subset files | commit them (`--allow-dirty` only for smoke tests) |
| warning `N/k prototypes have no members` | k too large, or many duplicate images | lower k |
| warning `N/10 spherical k-means restarts hit max_iter` | didn't converge | raise `max_iter` (changes the name: `-maxiter<n>`) |
| `--set 'selection.embedding.params={K:256,seed:0}'` matches nothing | without spaces OmegaConf parses keys `"K:256"`, `"seed:0"` | use `selection.embedding.params.K=256 selection.embedding.params.seed=0`, or `{K: 256, seed: 0}` with spaces |

---

## 13. Summary in one paragraph

`02_select.py` loads one embedding per training image (DINOv2-B CLS from `derived.parquet` by
default), L2-normalizes them, and runs spherical k-means (greedy k-means++ seeding, cosine
assignment, unit-length prototypes, 10 restarts seeded by `--seed`). Every image is scored by
cosine distance to its prototype and ranked — descending for `keep: hard`, ascending for
`keep: easy`, ties by row order. By default (`balance: 0.5`) each cluster is first guaranteed
`floor(0.5 × its fair share)` slots, and the rest of the `round(ratio × N)` budget goes to the
best-ranked images overall. The sorted ids go to
`results/subsets/<dataset>_<selection-name>_<ratio>_<seed>.json`, where the auto-generated
selection name (e.g. `proto-hard-k20-bal50-dinov2vitb14-cls`) encodes every subset-changing
setting. The file is committed and passed to `03_train.py`, which trains SegFormer-B0 for 100
epochs over the subset with the frozen recipe, evaluates with the full val protocol, and
appends mIoU and rare-class mIoU to `runs.csv`. `02b_inspect_prototypicality.py` runs the same
selection without writing a subset and, with `--k-sweep`, measures how much the subset
depends on k and on the k-means seed.
