# Proposals 1, 2, 3 explained with one toy example

A helper for `docs/method-proposals-2026-10-06.md`. It runs the same tiny dataset through
today's method and each proposal, so the differences are easy to see.

## The short version

Each proposal changes a **different part** of the selection pipeline:

| | What it changes | Question it asks | Unit picked | Does an image's value depend on what's already picked? |
|---|---|---|---|---|
| **Today** (patch-mean prototypicality) | — | "Does this image look unusual *on average*?" | whole image | No |
| **Proposal 1**: dense prototypicality | **the score** | "Does this image contain *a few* very unusual patches?" | whole image | No |
| **Proposal 2**: rarity-weighted coverage | **the selector** | "Does this image add something rare that the subset *doesn't have yet*?" | whole image | **Yes** |
| **Proposal 3**: region-level selection | **what gets labeled** | "Which *parts* of which images are worth labeling?" | image regions | Yes (budget is in pixels) |

- Proposal 1 still ranks images one at a time by a score, just a better one.
- Proposal 2 stops ranking. It builds the subset step by step and checks what's missing.
- Proposal 3 stops picking whole images at all.

---

## The toy dataset

Each image is cut into 100 patches (the real images have 37×37 = 1,369). Every patch belongs
to one "kind of stuff", called a concept.

| Image | Patches | In words |
|---|---|---|
| img1 | 50 road, 30 building, 18 sky, **2 traffic light** | ordinary street with a traffic light |
| img2 | 50 road, 32 building, 18 sky | ordinary street, nothing rare |
| img3 | 48 road, 32 building, 18 sky, **2 traffic light** | almost a copy of img1 |
| img4 | 40 road, 45 vegetation, 10 sky, **5 motorcycle** | park road with a motorcycle |
| img5 | 30 road, 20 building, 50 construction | construction site: odd-looking, but no rare class |

**Budget: keep 3 of the 5 images.**

The ideal subset is **img1, img4, img5**: it has the traffic light, the motorcycle, and every
other concept, with no duplicate.

---

## Today: patch-mean prototypicality

1. Average each image's 100 patch vectors into **one vector**.
2. Cluster those vectors; score each image by its distance to the nearest cluster center.
3. Keep the most unusual (farthest) images.

What happens:

- img5 is half construction, so its average is far from everything: **most unusual**.
- img4 is half vegetation: **unusual**.
- img1, img2, img3 have almost the same average. 2 traffic-light patches out of 100 barely
  move it, so they all look like the same ordinary street.

**Result:** img5, img4, then a **coin flip** among img1/img2/img3. If img2 wins, the subset
has no traffic light at all.

> This is the failure seen in the results: traffic light and pole keep losing IoU, because the
> images that contain them look ordinary as a whole.

---

## Proposal 1: dense prototypicality ("judge an image by its strangest patches")

1. Cluster **patches** (not images) into prototypes.
2. Score **every patch** by its distance to the nearest patch prototype. Common stuff scores
   low, rare stuff high.
3. An image's score is the **average of only its few highest-scoring patches** (the real
   version uses the top ~5%; here, the top 2 patches).
4. Keep the highest-scoring images.

Patch scores in the toy: road/building/sky 0.1, vegetation 0.3, construction 0.6,
traffic light 0.8, motorcycle 0.8.

| Image | Top-2 patches | Image score |
|---|---|---|
| img1 | 2 traffic light | **0.8** |
| img3 | 2 traffic light | **0.8** |
| img4 | 2 motorcycle | **0.8** |
| img5 | 2 construction | 0.6 |
| img2 | 2 building | 0.1 |

**Result:** img1, img3, img4.

- ✅ The traffic light is kept. A small rare object now counts, because the score looks only
  at the tail instead of the average.
- ❌ img1 and img3 are **near-duplicates** and both get picked. Each image is scored on its
  own, so the method can't tell it already has a traffic-light street. (In practice, cluster
  balancing softens this a little.)
- ❌ Construction (img5) is dropped.

**What it tests for the paper:** "Where should unusualness be measured?" It is the next step
on the line **CLS → patch mean → patch tail**. If results keep improving along that line, that
is the mechanism figure.

---

## Proposal 2: rarity-weighted coverage ("fill the sticker album")

1. Cluster patches into concepts (as above).
2. Give each concept a **weight that grows as it gets rarer**. Here: road/building/sky 0.1,
   vegetation/construction 0.5, traffic light/motorcycle 3.
3. Build the subset **one image at a time**. At each step, add the image that brings the most
   **weight of concepts not yet in the subset**.

| Step | Gain of each candidate (only *new* concepts count) | Pick |
|---|---|---|
| 1 | img1: road+building+sky+traffic light = 3.3 · img3: 3.3 · **img4: road+veg+sky+moto = 3.7** · img5: 0.7 · img2: 0.3 | **img4** |
| 2 | **img1: building+traffic light = 3.1** · img3: 3.1 · img5: building+construction = 0.6 · img2: 0.1 | **img1** |
| 3 | img3: **0** (nothing new) · **img5: construction = 0.5** · img2: 0 | **img5** |

**Result:** img4, img1, img5. That's the ideal subset.

- ✅ The traffic light is kept, and the near-duplicate img3 is **skipped**: once img1 is in,
  img3 adds nothing new.
- ✅ Construction is kept too, because it's the only uncovered concept left.

The key difference from Proposal 1: **the same image is worth a lot or nothing depending on
what's already in the subset.** img3 scored 3.3 at step 1 and 0 at step 3.

Two refinements in the real version:

- **Saturation (τ):** 2 traffic-light patches don't fully "cover" traffic lights, because one
  example isn't enough to learn a class. So img3 would still earn a little at step 3, just much
  less than at step 1.
- **Hybrid (λ):** add a bit of today's patch-mean unusualness to each gain, to keep the
  overall-mIoU gain prototypicality already has.

**What it tests for the paper:** this is Contribution 4, "the fix". The decisive comparison is
against `zcore`, which uses the same coverage idea on whole-image embeddings.

---

## Proposal 3: region-level selection ("label parts of images, not whole images")

The first two proposals both assume an annotator labels **every pixel** of each chosen image.
In Cityscapes most of those pixels are road, building and sky, which the model learns easily.

Proposal 3 changes the **budget** from "3 images" to "**300 labeled patches**" (the same
amount of annotation work) and lets you spend it anywhere:

| | Images vs regions labeled | Traffic-light patches labeled | Motorcycle patches labeled |
|---|---|---|---|
| Proposal 2 (whole images img4, img1, img5) | 3 full images = 300 patches, ~150 of them road/building/sky | 2 | 5 |
| Proposal 3 (regions) | e.g. the traffic-light regions of img1 *and* img3, the motorcycle region of img4, the construction area of img5, plus a sample of road/building/sky for context, and the same from many *other* images | many more | many more |

In the real dataset the effect is large: instead of 300 images' worth of road, you can label
traffic lights from *thousands* of images.

Training then ignores the unlabeled pixels (they're set to `ignore_index`, so they add no
loss).

- ✅ The biggest expected gains on rare classes, because the budget stops paying for common
  stuff.
- ❌ The most work: the trainer and dataset need per-image region masks.
- ❌ It changes the setting from "how many images to label" to "how many pixels to label".
  Baselines must change to match: random images vs random regions at the same pixel budget.

**What it tests for the paper:** whether *choosing whole images* is itself the limit
(mechanism (b) in plan §14b). If Proposals 1 and 2 still can't recover traffic light and
pole, Proposal 3 is the next step.

---

## How the three fit together

```
            Proposal 1                 Proposal 2                 Proposal 3
  change:   how an image is SCORED     how images are COMBINED    what UNIT is labeled
  fixes:    rare things hidden by      duplicates; rare things    paying to label road
            averaging                  never protected            and sky over and over
  still     duplicates; ranks images   still labels whole         —
  misses:   independently              images
```

They can be combined. For example, Proposal 1's patch scores can set Proposal 2's concept
weights, and Proposal 2's coverage objective can choose regions in Proposal 3.

## What each predicts in the results

| | Overall mIoU vs today | Traffic light / pole / sign | Motorcycle / train (unusual scenes) |
|---|---|---|---|
| Proposal 1 | about the same or better | **recovers part** of the loss | stays good |
| Proposal 2 | about the same | **recovers more** (duplicates removed, rare concepts protected) | stays good |
| Proposal 3 | not directly comparable (different budget) | **largest gain** | gain |

If a proposal doesn't move the traffic light / pole / sign columns, it hasn't fixed the
problem the results identified, whatever it does to overall mIoU.

---

# Designing the experiments

The new proposals have many knobs, but most of them should **not** be varied by training.
Three rules keep the design small and comparable to Experiments 3–9:

1. **The training side stays exactly as before.** Same model, recipe (`cityscapes_proxy`,
   lr 1e-4, 100 epochs), eval protocol, ratios and seed meaning. Every new knob is on the
   *selection* side, so any difference in results comes from *which images* were picked, just
   as in Experiments 4–9. (Proposal 3 is the one exception; see below.)
2. **Each proposal changes one thing relative to the current best method, and only that
   thing gets trained.** Every other knob is either inherited from earlier experiments or fixed
   to a principled default *before* any results are seen.
3. **Selection is cheap and training is expensive, so tune offline.** Making a subset takes
   seconds or minutes; training it takes hours. Choose among knob values with a label-free
   check that needs no training (below). Train only the 1–3 finalists per proposal.

## Fixed reference rows (compare everything against these)

| Row | Source | Role |
|---|---|---|
| `random` | Exp 3, plus seeds 3 and 4 at 0.1/0.2/0.3 (reassessment item 0b) | The band every method must clear |
| patch-mean proto, k50, bal50 | Exp 9 | **The current best.** A new proposal must beat this, not just random |
| CLS proto, k10, bal50 | Exp 4 | The global-embedding reference, for the CLS → patch → tail story |
| `zcore` | reassessment item 1b | The strongest global coverage method; the decisive foil for Proposal 2 |

If the features are re-extracted without the 4× squash (reassessment item 1c), **rerun the
patch-mean k50 row on the new cache** (9 runs) and build all three proposals on that cache.
Otherwise a gain could come from the better features, not from the proposal.

## The comparison standard

- **Ratios 0.1 / 0.2 / 0.3, seeds 0 / 1 / 2** (9 runs, about 3 GPU-h per configuration).
  This is where the existing methods separate. At 0.5 and 0.7 every method is within about
  1 point of random.
- **Pre-screen at 0.2 only** (3 runs, about 1 GPU-h) when choosing between two or three
  variants of the same proposal. Only the winner gets 0.1 and 0.3.
- **The full ladder (add 0.5 / 0.7)** only for the final method that goes in the paper.
- **Seed meaning stays the same:** seed N sets the selection randomness (k-means starts) and
  the training RNG.
- **Naming:** give each new selector `name: auto`, encoding every knob that changes the subset
  (as `proto-hard-k50-bal50-...` does). Then subsets, `runs.csv` rows and the W&B report
  separate the variants automatically.
- **Report the same columns as now,** plus per-class IoU for traffic light, pole, sign,
  motorcycle and train. Those five classes are where the proposals are supposed to differ.

## The label-free check (used to choose knob values without training)

For every candidate subset, compute **rare-concept coverage** offline:

1. Fit a patch vocabulary (spherical k-means on the shared patch sample, reassessment item 1d).
   Call the rarest 25% of concepts (by patch count) the *rare concepts*.
2. For the subset, count how many rare concepts have at least m = 20 patches in it, and what
   fraction of all rare-concept patches it contains.

Two safeguards:

- **Avoid circularity.** Proposal 2 directly maximizes coverage of its own vocabulary, so
  measure the check on a *different* vocabulary (different K, e.g. 1024, and a different seed).
  Otherwise Proposal 2 wins the check by construction.
- **Validate the check on results you already have.** Compute it for random, CLS k10/20/50 and
  patch-mean k10/20/50 at 0.1/0.2/0.3, and see whether it orders them the way their *trained*
  rare-class mIoU does. If it doesn't, don't use it to choose knobs: fall back to the 0.2
  pre-screen. (This uses validation-set results only to check the check, never to pick a
  method's knobs, so selection stays label-free.)

Class composition of each subset (per-class pixel counts from the labels) is reported next to
it **as analysis only**, never to choose knobs.

---

## Proposal 1: dense prototypicality

**The one thing that changes:** the score. Image distance-to-prototype becomes the mean of the
image's top-q% patch distances.

The existing selector already separates scoring from picking:
`select_by_score(scores, clusters, k, budget, keep, balance)` in
`segcoreset/selection/prototypicality/`. So **reuse the exact same picking step and the same
image clusters as patch-mean k50**, and swap only the `scores` array. Then the experiment
differs from Exp 9 in exactly one thing.

| Knob | Decision | Why |
|---|---|---|
| keep, balance | **inherited**: hard, 0.5 | Same as Exp 4–9 |
| image clusters k (used **only** for balancing) | **inherited**: patch-mean k50 clusters | Only the score changes |
| patch vocabulary size K_p (used **only** for scoring) | **fixed**: 256, sensitivity {64, 1024} checked offline | See below |
| patch sample (used **only** to fit the vocabulary) | **fixed**: 200 patches per image, seed 0 | See below |
| **tail fraction q** | **the variable**: candidates 100% (all patches), 20%, 5%, 1% | It *is* the proposal: how local the score is |

### The two clusterings and the patch sample, explained

**Proposal 1 uses two separate clusterings, for two different jobs.** In today's method one
k-means over image embeddings does both jobs. Proposal 1 splits them apart:

| | Image clusters (k = 50) | Patch vocabulary (K_p = 256) |
|---|---|---|
| What is clustered | Images (one patch-mean vector per image, 2,975 points) | Patches (individual 14×14 tokens) |
| Job | **Balancing**: how the budget is spread across groups of similar scenes (`balance = 0.5`) | **Scoring**: how unusual each patch is |
| Where it comes from | Already computed: the same k50 assignment as Exp 9 (patch-mean k50) | New: fitted once on the patch sample |
| Changes in Proposal 1? | No | Yes, this is the new part |

So K_p = 256 does **not** replace k = 10/20/50. The k = 50 image clusters still decide how many
images each scene group contributes. The 256 patch prototypes only decide which images inside
a group rank as "hard". Keeping k = 50 means the only difference from Exp 9 is the score. If
you also changed k, a gain or loss couldn't be attributed to the score. (k = 50 is chosen
because it is the strongest existing patch-mean row. You don't need to rerun P1 at k = 10/20:
varying k is not what Proposal 1 tests.)

The full pipeline:

```text
A. once:      200 random patches from each image  ->  ~595k patch vectors
              spherical k-means (K_p = 256, several starts)  ->  256 patch prototypes
B. per image: for ALL 1,369 patches:  d = 1 - max cosine(patch, prototypes)
              image score = mean of the top q% of d   (q = 5% -> top 69 patches)
C. pick:      select_by_score(scores = B, clusters = patch-mean k50 labels, k = 50,
                              budget, keep = hard, balance = 0.5)      # unchanged code
```

**Why K_p = 256, and how to justify it.** Be honest in the paper: 256 is not tuned. It is the
default of the existing `kmeans_hist` derivation (`scripts/00b_derive_embeddings.py`,
`--k 256`, plan §5.3). The arguments for it:

- **It must be much finer than the 19 classes.** One class covers many patch appearances
  (near and far car, car roof vs wheel, lit vs shadowed road). About 13 prototypes per class is
  a reasonable granularity.
- **It must be fittable.** With ~595k sample patches, 256 prototypes get about 2,300 patches
  each on average, so the centroids are stable.
- **K_p matters in both directions, so check it.** If K_p is too small, rare content (traffic
  light) is merged into a common prototype (pole, building edge). Then it sits far from that
  prototype's center, which is what we want, but so does every odd-looking common patch, so
  the tail gets noisy. If K_p is too large, rare content gets **its own prototype**, sits close
  to it, and looks *typical*. The score then hides exactly what it should find. Somewhere in
  between, rare content is poorly fitted and common content is well fitted.

How to defend the choice without training many runs: compute the P1 scores for
K_p ∈ {64, 256, 1024} and report (a) the Spearman correlation of the image rankings and
(b) the label-free rare-concept check for each. If the rankings agree closely (say ρ > 0.9),
write "results are insensitive to K_p in [64, 1024]" and keep 256. If they don't, K_p is a real
knob: pick it with the label-free check, and train the runner-up only if the check is close.
Also log the cluster sizes; prototypes with almost no patches are a sign K_p is too large.

**Why sample 200 patches per image, and why that number.** An image does have far more patch
tokens than 200. At the current 518×518 input, DINOv2 ViT-B/14 gives a 37×37 grid, so **1,369
patches per image**. After the aspect-preserving re-extraction (518×1036, reassessment 1c) it
would be 37×74 = 2,738.

The 200-patch sample is used **only to fit the 256 prototypes (step A)**. Scoring (step B) uses
**all** patches of every image. The sample exists for two reasons:

1. **Memory.** All patches are 2,975 × 1,369 ≈ 4.1M vectors × 768 floats ≈ 12.5 GB in float32
   (≈ 25 GB after re-extraction). Spherical k-means with several starts on that is slow and may
   not fit in RAM. 200 per image gives ~595k vectors ≈ 1.8 GB, about 15% of all patches.
2. **The audit #4 fix.** The current `kmeans_hist` fitting streams images one at a time in city
   order, so the first image and the early cities shape the clusters. Fitting on a shuffled
   sample that contains every image fixes this.

200 is a memory-driven choice, not a tuned value. It was picked so the sample fits easily in
memory and every image contributes the same number of patches. Justify it with a stability
check, not with theory: refit the vocabulary with a different sample seed and with 400 patches
per image, and report the rank correlation of the resulting P1 image scores. If ρ stays high,
the sample size doesn't matter.

How to draw the sample: flatten each image's patch grid to [1369, 768], take
`randperm(1369, generator seeded by (seed, image index))[:200]`, i.e. uniformly without
replacement, and concatenate over all images. Uniform sampling keeps tiny objects at their real
frequency in the sample. That is fine for Proposal 1: if rare content is underrepresented in the
vocabulary, it gets no prototype of its own and keeps a high distance, which is what the tail
score wants. (It matters more for Proposal 2, whose concepts come from the same vocabulary;
the rarity weight is meant to correct for it there.)

**Note on q:** q is a fraction of the image's *full* patch grid, not of the 200 sample:
q = 5% is the top 69 of 1,369 patches (ceil of 68.45), q = 1% the top 14. Keep q as a fraction so it stays
comparable if the grid grows after re-extraction. Fit the vocabulary and the k50 clusters on
the **same feature cache** as the Exp 9 baseline; if you re-extract (1c), rerun patch-mean k50
on the new cache too, so the comparison stays one-change.

Plan:

0. Offline only: the K_p ∈ {64, 256, 1024} and sample-seed / 400-patch stability checks above.
1. Compute the label-free check for q ∈ {100, 20, 5, 1}% at 0.1/0.2/0.3. No training.
2. Train **2 configurations**: q = 5% (the proposal's default), plus q = 100% *or* whichever
   value the check prefers, if different. q = 100% is the "same patches, no tail" control.
   9 runs each, **18 runs, about 6 GPU-h.**
2b. **q sweep** (added 2026-10-11): also train q = 1%, 20% and 50%, giving five values
   {1, 5, 20, 50, 100}% (14 / 69 / 274 / 685 / 1,369 patches). They are roughly log-spaced and
   chosen so neighbouring values change the 0.1 subset by similar amounts (Jaccard 0.58, 0.69,
   0.73 between neighbours). Finer steps share at least 80% of their images with a neighbour, so
   they would mostly retrain the same subset. Below 1% the score rests on a handful of patches;
   add 0.5% only if 1% wins. **+27 runs, about 9 GPU-h** (45 in total for P1). The result is a
   curve of mIoU and per-class IoU against q, with Exp 9 and random as horizontal lines.
3. Optional, same setup: **Idea A2** (rarity-weighted pooling) as one more score. It sits
   between patch mean and tail on the locality line. 9 runs.

**Outcome table** (one row each, Δ vs random *and* vs patch-mean k50):
CLS k10 → patch-mean k50 → [A2] → P1 q = 100% → P1 q = 5%. If mIoU and traffic light / pole
improve down the rows, that is the mechanism figure.

## Proposal 2: rarity-weighted coverage

**The one thing that changes:** the selector. Greedy coverage replaces score-and-rank.
Because the selector changes, there is no inherited k or balance; the comparison is at equal
ratio against the reference rows.

| Knob | Decision | Why |
|---|---|---|
| vocabulary K | **fixed**: 256 (the same vocabulary as P1) | One shared fit |
| saturation τ | **set from data, not tuned**: per concept, τ_k = median patch count of concept k in images that contain it | Scale-free: a concept that is usually small (traffic light) saturates after a few patches, a large one (road) after many. No knob left |
| **rarity exponent α** | **the variable**: 0 (rarity off) vs a default chosen from {0.5, 1} by the check | "Rarity on/off" is the ablation the plan asks for (§10, Stage 3) |
| hybrid weight λ | **0 first**; one λ > 0 only as a follow-up | λ = 0 is pure coverage, the clean test. Before mixing, rescale both terms to similar ranges (e.g. rank-normalize atypicality to [0, 1] and multiply by the mean greedy gain) so λ means "fraction of the gain" |

Plan:

1. Compute the label-free check for α ∈ {0, 0.5, 1} (on the *independent* vocabulary). No
   training.
2. Train **α = 0** and **α = default**, both with λ = 0. **18 runs, about 6 GPU-h.**
3. If the α = default run beats random but loses overall mIoU to patch-mean k50 (the expected
   trade-off), train **one hybrid** with λ chosen by the check. 9 runs, about 3 GPU-h.

**Outcome table:** `zcore` (global coverage) vs P2 α = 0 (dense coverage, no rarity) vs P2
α = default (dense + rarity) vs patch-mean k50. That table separates "dense helps" (`zcore` →
α = 0) from "rarity helps" (α = 0 → α = default).

## Proposal 3: region-level selection

This is the one proposal that **changes the training side**, so it needs two extra decisions
to stay comparable.

**Decision 1: the budget axis.** Define the budget as the **fraction of all training pixels
that get labels**. Labeling 20% of the images is labeling 20% of the pixels, so every earlier
result is already a point on this axis (`random` at 0.2 = "20% of pixels, chosen as whole
images"). No new x-axis is needed.

**Decision 2: compute.** The trainer sets steps from the number of images
(`len(train_ds) // batch_size`, ADR 0001). Spreading 20% of the pixels over, say, 60% of the
images would quietly give 3× the gradient steps, and the comparison would be unfair. **Match
steps to the image-level run at the same ratio** by setting `epochs` per run so that
`epochs × (images touched // batch_size)` equals the image-level step count. Read ADR 0001
before doing this, and record it in the run config.

| Knob | Decision | Why |
|---|---|---|
| region unit | **fixed**: square grid cells (e.g. 128×128 px on the 1024×2048 image) | Simplest unit, no superpixel algorithm to defend; aligns with the patch grid |
| unlabeled pixels | **fixed**: `ignore_index` (no loss) | Standard partial-label training |
| region score | **inherited**: P1's patch scores averaged per cell, or P2's coverage gain run over cells, whichever won | Reuse; don't invent a new scorer |
| steps | **matched** to the image-level run (Decision 2) | Fair compute |
| **selection unit** | **the variable**: random cells vs scored cells (vs random images = Exp 3) | Separates "regions beat images" from "my scorer beats random" |

Plan:

1. Train **random cells** (new baseline) and **scored cells** at 0.1/0.2/0.3 × 3 seeds.
   **18 runs, about 6 GPU-h** at matched steps, plus about 1 day of trainer and dataset work for
   per-image label masks.
2. Compare three rows on the same pixel-budget axis: random images (Exp 3) → random cells →
   scored cells.
   - random images → random cells shows the value of labeling regions instead of images.
   - random cells → scored cells shows the value of the selector.

**One caveat to check:** random 512×1024 training crops can land on mostly unlabeled areas,
which wastes steps. Log the fraction of labeled pixels per batch. If it's low, sample crops
around labeled cells, and use the same crop rule for both P3 rows.

---

## Order and total cost

| Step | Runs | GPU-h | Depends on |
|---|---|---|---|
| Shared setup: patch sample + vocabulary + label-free check, validated on the existing subsets | 0 | 0 (CPU, ~1 day of code) | reassessment 1c/1d |
| Proposal 1 (q = 5% and its control) | 18 | ~6 | setup |
| Proposal 1 q sweep (q = 1%, 20%, 50%) | 27 | ~9 | P1 core pair |
| Proposal 2 (α = 0, α = default; maybe a hybrid) | 18–27 | ~6–9 | setup, `zcore` |
| Proposal 3 (random cells, scored cells) | 18 | ~6 | the winner of P1/P2 as its scorer |
| Final method on the full ladder (0.5 / 0.7) | 6 | ~5 | all of the above |
| **Total** | **~60–70** | **~25–30** | |

Run them in the order **P1 → P2 → P3**. Each one reuses what the previous one built: P1 needs
the patch vocabulary, P2 reuses the vocabulary and P1's scores (for the hybrid), and P3 reuses
whichever scorer won.

## Write the success criteria down before running

So results can't be reinterpreted afterwards, decide now what counts as a win:

- **P1 works** if, at ≥ 2 of the 3 ratios, it is above patch-mean k50's ±1 std on mIoU, *or* it
  recovers at least half of the traffic light / pole / sign loss without losing mIoU.
- **P2 works** if it beats `zcore` at ≥ 2 of 3 ratios on rare-class mIoU, and α = default beats
  α = 0 (otherwise the rarity weighting isn't what helps).
- **P3 works** if scored cells beat random cells (the selector matters), and random cells beat
  random images (the unit matters). Each answers a different question, so report both even if
  only one holds.
