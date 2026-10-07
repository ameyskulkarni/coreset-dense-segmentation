# Method proposals after the prototypicality batch (2026-10-06)

Three candidate methods to run next, alongside the planned `patch_mean` prototypicality batch
(3 k × 5 ratios × 3 seeds = 45 runs). Written after Experiments 1–7 and the 2026-10-06
analysis in `docs/results_wandb.md`. Nothing here has been implemented or run yet.

## Starting point: what the results so far say

- Keep-hard CLS prototypicality beats random on **overall mIoU** by about +0.5–1.5 points at
  20–70% kept, consistently across k = 10/20/50. The gain disappears at 10%.
- It gives **no robust rare-class gain**. The k = 10 rare-class win didn't replicate at
  k = 20/50.
- The per-class picture is a **trade-off**:
  - Classes that live in visually unusual scenes gain (motorcycle 12/15 cells positive,
    fence 15/15, wall 13/15).
  - Thin "street furniture" classes that live in ordinary scenes lose (traffic light 0/15,
    pole 0/15, traffic sign 3/15). Traffic light collapses from 32.4 to about 5 IoU at 10%.

**Interpretation:** a global (whole-image) score decides which *scene types* are kept.
Keep-hard removes ordinary street scenes first, and those scenes hold most of the traffic
lights and poles. Atypicality is being measured at the wrong granularity for segmentation.
All three proposals below fix that in different ways.

## Caveat on the `patch_mean` batch

`patch_mean` mean-pools the 37×37 patch grid into one global vector, so it averages away
the local regions the thesis is about. Its subsets may look a lot like the CLS subsets.

**Cheap check before spending about 23 GPU-h:** compute the Jaccard overlap between the
`patchmean` and `cls` subset files at each (k, ratio). This takes minutes and needs no
training. If the overlap is above about 0.7, the 45 runs will mostly repeat Experiments 4–7.
The batch is still useful as the middle step of Proposal 1's ablation (CLS → patch mean →
patch tail), possibly at reduced size.

---

## Proposal 1: Dense prototypicality (atypicality per region, not per image)

**Recommended to run in parallel with `patch_mean` first.**

### Idea

Keep Sorscher-style prototypicality, but apply it to patch tokens instead of the CLS token:

1. Sample about 200 patches per image (fixed seed), shuffle, and run spherical k-means with
   K ≈ 100–256 to get *patch prototypes*.
2. Score every patch by its cosine distance to the nearest patch prototype.
3. Score each image by a **tail aggregate** of its patch scores: the mean of its top-q%
   hardest patches (q ≈ 5%), *not* the mean over all patches.
4. Keep the hardest images, with the same cluster balancing as the current preset.

### Why it's novel and fits the paper

- It gives a clean ablation with one variable, where atypicality is measured:
  **CLS → patch mean → patch tail**. That is a direct test of mechanism (c) in plan §14b,
  "global embeddings average away the local regions that carry rare-class signal".
- It makes a prediction that can fail: patch mean ≈ CLS, while patch tail recovers traffic
  light and pole. An ordinary intersection with a traffic light now contains a few atypical
  patches, so it stops being pruned first. If this holds, the per-class table becomes the
  mechanism figure.

### Cost

- Code: mostly reuse. Spherical k-means already exists in the prototypicality `scoring.py`,
  and the patch cache already exists.
- Training: screen at ratios 0.2 and 0.3 only, 3 seeds = 6 runs, about 2.3 GPU-h. These are
  the ratios where prototypicality showed signal. Audit #1 shows 0.1 is dominated by the
  training budget rather than by subset content.

---

## Proposal 2: Rarity-weighted patch-concept coverage (+ hybrid with prototypicality)

This is the plan's `patch_coverage` (§5.3, Contribution 4), refined using the results above.

### Plain-language explanation

Cut every image into small squares (patches), sort the patches into "kinds of stuff", then
pick images so the subset contains **at least a few examples of every kind of stuff**, with
priority on the rare kinds.

1. **Cut images into patches.** DINOv2 already splits each image into a 37×37 grid
   (about 1,369 patches). Each patch has a feature vector describing what it looks like.
   These are cached.
2. **Group similar patches into "concepts".** Run k-means over patches from all images
   (e.g. K = 256). Without any labels, the groups come out like "road surface",
   "tree leaves", "looks like a traffic light", "motorcycle-ish". These are
   **pseudo-classes**.
3. **Describe each image by its concepts.** Count how many of each image's patches fall in
   each concept, e.g. image A = 400 road, 300 building, 3 traffic light.
4. **Weight concepts by rarity.** Road patches number in the millions, motorcycle-like
   patches in the thousands. Give rare concepts a large weight and common ones a small
   weight. This is the "rarity-weighted" part.
5. **Pick images greedily, like filling a sticker album.** Every concept is a sticker, and
   rare stickers are worth more. After the first pick, an image only earns points for
   concepts the subset *doesn't have enough of yet*. The 500th road image earns almost
   nothing; the first motorcycle image earns a lot. At each step, add the image that earns
   the most new points, until the budget is reached. This is the "coverage" part.

**Why this addresses the trade-off:** the method doesn't care whether the *whole image*
looks unusual. An ordinary street scene with a few traffic-light patches still earns points
for them, as long as the subset is short on traffic lights. Rare things are protected even
inside boring images.

**Tiny example** (pick 2 of 3 images):

| Image | Contents |
|---|---|
| A | road, building, sky |
| B | road, building, sky, plus a small traffic light |
| C | road, trees, plus a motorcycle |

- Keep-hard prototypicality: picks C plus A or B, depending on which looks odder overall.
  Whether the traffic light survives is a coin flip.
- Coverage: picks C first (rare motorcycle and trees). Then B beats A, because B adds the
  traffic light and A adds nothing new. Result {C, B}: every concept is covered.

### Formal objective

```
F(S) = Σ_k  w_k · [ 1 − Π_{i∈S} (1 − p_ik) ]
p_ik = 1 − exp(−n_ik / τ)        # saturating: more patches of concept k → closer to 1
w_k  = freq_k^(−α)               # rarity weight; α = 0 turns weighting off
```

- `n_ik` = number of image i's patches in concept k; `freq_k` = concept k's share of all
  patches.
- `F` is monotone submodular, so greedy selection has the 1 − 1/e guarantee and runs in
  seconds for 2,975 images × 256 concepts (use lazy greedy).
- **Saturation (τ):** one traffic-light patch isn't enough to learn traffic lights, so an
  image counts more toward a concept the more of its patches belong to it. τ sets how many
  patches count as "plenty".
- **Hybrid:** add λ · (CLS atypicality) to each image's greedy gain. This keeps
  prototypicality's robust overall-mIoU gain while the coverage term protects the
  street-furniture concepts that keep-hard drops.
- Choose α, τ, λ and K with a **label-free proxy** (plan §9), not by training.

### Why it's the paper's payoff

It is Contribution 4. The decisive comparison is against `zcore`, the strongest global
coverage method (plan §5.1a, Appendix A). The per-class finding gives a specific prediction:
coverage should beat prototypicality on traffic light, pole and sign while matching it on
motorcycle.

### Prerequisite: fix `kmeans_hist` (audit #4)

Steps 2–3 already exist as `_kmeans_hist` in `scripts/00b_derive_embeddings.py:66`. Steps 4–5
don't exist yet (a new selector, e.g. `segcoreset/selection/patch_coverage.py`). Today the
only consumer of `kmeans_hist` is prototypicality, which treats it as one more embedding.

The current fitting is broken:

```python
km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=8192, n_init=3)
for i in ids:                      # sorted: aachen_... first, zurich_... last
    km.partial_fit(<1,369 patches of one image>)
```

1. All K starting centers are placed from the **first image only**
   (`aachen_000000_000019`), which can't seed groups for content it doesn't contain.
2. Images stream **in city order** and per-center update sizes shrink over time, so early
   cities dominate the groups and later cities barely move them.
3. `n_init=3` is ignored by `partial_fit` (single start), and `batch_size=8192` does nothing
   because each call gets one image.

Effect on coverage: rare content gets merged into common groups, never receives its own high
rarity weight, and can't be protected. A failure would then look like "dense coverage
doesn't work" when the clustering was the cause.

**Fix** (only the fitting part changes; the histogram part, lines 86–91, stays):

1. Sample about 200 patches per image with a fixed seed (~600k × 768, ~1.8 GB float32).
2. Shuffle and fit on the whole sample with several starts. Prefer spherical k-means, since
   patches are L2-normalized.
3. Assign all patches to the fitted centers as now; log cluster sizes (empty or near-empty
   clusters).
4. Record the sample seed and sample size in `params`, so the new derivation gets a distinct
   name. Check whether any committed subset was built from the old `kmeans_hist` first.

Proposal 1's patch sample is the same step, so the two proposals share it.

---

## Proposal 3: Region-level selection at a matched labeled-pixel budget

**Highest upside, most engineering.**

### Idea

Change what the budget counts: **labeled pixels instead of images**. Select (image, region)
pairs, such as superpixels or patch blocks, by the rarity of their patch concepts. Train with
the unlabeled pixels masked out (`ignore_index`). Compare against random images and random
regions at equal labeled-pixel budgets.

### Why it's novel

- Region-based active learning exists (superpixel AL, RIPU, ...), but it is warm-start and
  uncertainty-based. A **label-free, one-shot, cold-start region selector** is largely open.
- It tests mechanism (b) in plan §14b head-on: if image-level selection can't isolate rare
  classes, stop selecting images.
- Gains are likely large, because the budget stops paying to label road and sky in every
  kept image.

### Cost

- Trainer: masked loss plus per-image region masks in the dataset. About a day of work.
- It changes the framing (labeling cost per pixel, not per image), so it fits better as a
  second paper section, or as the "fix" if Proposals 1 and 2 only partly help.

---

## Brainstorm ideas (added 2026-10-07)

Two further ideas, discussed after the three proposals above. Neither is implemented.

Facts about the current feature cache that constrain both (checked in
`segcoreset/features/dino.py`):

- The backbone is `dinov2_vitb14` **without registers**. Each cached patch vector is
  **L2-normalized**, so patch norms are lost; anything norm-based needs re-extraction.
- **Attention maps are not cached.** Attention-based weighting also needs re-extraction.

### Idea A: Weighted patch mean instead of the plain mean

**Key point.** The CLS token is already a weighted average of the patches, with weights
learned by DINOv2's attention (which favors the main salient object). A weighted patch mean
is therefore a point on a spectrum:

```
uniform mean (patch_mean)  ←  custom weighting  →  CLS (learned attention weights)
```

The real question is whether a hand-chosen weighting emphasizes what segmentation needs
(small, rare things) better than DINOv2's learned weights.

**Weighting options** (most promising first):

| # | Weighting | What it does | Re-extraction? |
|---|---|---|---|
| A1 | **Concept-balanced pooling**: average patches within each concept, then average the concepts | Every kind of stuff in the image gets one equal vote regardless of area; a traffic light counts as much as the road | No (needs the fixed patch k-means) |
| A2 | **Rarity (IDF) weighting**: weight = 1 / frequency of the patch's concept | Like TF-IDF in text retrieval: common stuff (road, sky) muted, rare stuff amplified | No (same) |
| A3 | **Atypicality softmax**: weight ∝ exp(patch distance to nearest prototype / T) | One knob T: large T → plain mean, small T → only the strangest patches. A smooth version of Proposal 1's tail score | No |
| A4 | **GeM pooling** (generalized mean, power p) | p = 1 → mean, larger p → max. Standard in image retrieval | No |
| A5 | **Position prior**: down-weight bottom rows (ego-car hood) and top rows (sky) | Removes content shared by every image | No, but Cityscapes-specific and hard to defend |
| A6 | **Attention or norm weighting** | Close to what CLS already does | Yes |

**Assessment.**

- **On its own: low novelty.** IDF weighting (classic visual-words retrieval) and GeM pooling
  are well known. A reviewer would see "a better pooling function", i.e. one ablation.
- **As part of the paper: useful.** A1–A3 extend the representation spectrum
  (CLS → mean → concept-balanced / rarity-weighted → tail) for Contribution 3, and test
  directly *where in the image* atypicality should be measured, the mechanism story.
- **May not fix the core problem.** Any weighting still produces one vector and one score
  per image. The per-class trade-off seems to come from ranking whole images by one score;
  a better embedding moves the trade-off rather than removing it. Expect A2/A3 to recover
  some traffic-light/pole loss; A1 likely helps least because it ignores rarity.
- **Risk:** a mean over a few rare patches is noisy (rare concepts differ from each other),
  so distances may mostly measure "contains *something* odd" (glare, blur, construction)
  rather than "contains a rare class".

**Recommendation.** Keep as an ablation axis, not the headline. Screen A2 and A3 with the
screening harness below. A3 at one temperature is essentially Proposal 1, so fold it in as
Proposal 1's soft version. **A2 is the strongest stand-alone variant**: simple, easy to
explain, and aimed directly at the street-furniture loss.

### Idea B: Jigsaw-shuffling the image

Shuffling tiles destroys scene layout (road at the bottom, sky at the top) while keeping
local content. Four ways to use it, weakest to strongest:

**B1. Use the shuffled image's embedding for selection.**
- Reasonable intuition: Cityscapes scenes share one layout, which probably dominates CLS
  similarity; a shuffled embedding would reflect *content* more than *arrangement*.
- Problems: `patch_mean` is already order-insensitive; shuffled images are out of
  distribution for DINOv2 and tile seams create fake edges; results depend on the random
  permutation (needs averaging over several); tile size changes everything (14 px tiles
  destroy objects, 4×8 tiles keep them).
- **Verdict: probably weak**, hard to justify over `patch_mean`.

**B2. Jigsaw sensitivity as a score (the most original version).**
- Embed the image normally and shuffled; score = how much the embedding changes.
  - Small change → mostly texture/"stuff" (road, vegetation, facades).
  - Large change → meaning depends on arrangement (objects, structure, small things in
    context).
- **Per-patch version:** compare each patch's token in the original vs. the shuffled image,
  i.e. how much the patch relies on its surroundings. Hypothesis (untested): small objects
  like traffic lights are recognized partly via context (pole, sky), so they are highly
  context-dependent.
- Novelty: not aware of prior work using jigsaw sensitivity as a coreset score, but the
  literature has **not been searched yet**; check before claiming.
- Risk: speculative; "context-dependent" may not align with "useful for segmentation".
  Cheap to check.

**B3. Puzzle difficulty as a label-free hardness score.** Train a small jigsaw solver
(Noroozi & Favaro 2016) and score images by solving error. Needs an extra trained model and
the link to segmentation value is weak. **Not worth pursuing.**

**B4. The real lesson: an image is a bag of tiles.** If shuffling tiles keeps the useful
information, the image is the wrong selection unit: **select tiles, not images**. That is
Proposal 3 (region-level selection at a matched labeled-pixel budget). Use the jigsaw view
as the motivation for Proposal 3 in the paper; this is where the largest gains are expected,
at the highest engineering cost.

**Recommendation.** Skip B1 and B3. Run B2 through the screening harness as a speculative
score. Use B4 as the story behind Proposal 3.

### Screening harness (check ideas without training)

One offline script (under `scripts/experiments/`), about an hour per variant, no GPU
training:

1. **Overlap:** Jaccard of each variant's subsets with the CLS subsets at 0.2 and 0.3. If
   above about 0.7, the variant is effectively CLS; skip it.
2. **Subset composition (labels for analysis only):** per-class image and pixel counts in
   the subset vs. random, focusing on traffic light, pole, sign, motorcycle, train, bus,
   truck. Training results already show composition drives per-class IoU, so this predicts
   the direction of the result.
3. **Label-free proxy** (plan §9): coverage of rare patch concepts by the subset. **Method
   choices rest on this**, not on step 2, so labels don't leak into the method's design.
   Use step 2 only to show the proxy tracks real class composition, and report that.
4. **B2 only:** correlate jigsaw sensitivity with whether an image contains rare classes
   (analysis only).

Only variants passing steps 1–3 get the training screen (0.2/0.3 × 3 seeds = 6 runs).
Variants A1–A3 and the step-3 proxy depend on the `kmeans_hist` fitting fix (audit #4).

---

## Practical notes

- **Running in parallel on one 3090:** two runs fit in memory (~6.6 GB each) but slow each
  other down, so `gpu_hours_train` from concurrent runs is not comparable to Experiments 1–7.
  GPU-hours are a reported result (ADR 0001), so tag those rows or re-time one run per
  config later.
- **Rare-class metric:** report the image-rare set (audit #5) next to the frozen pixel-rare-5
  set, plus the per-class table. Traffic light and rider appear in more than half of all
  images, which can hide or exaggerate the effects of Proposals 1 and 2.
- **Aspect ratio (audit #6):** `segcoreset/features/dino.py:57` squashes 1024×2048 images to
  518×518, compressing thin objects 4× horizontally. This biases all three proposals against
  the dense representation. Re-extract at 518×1036 (37×74 grid) before Proposal 2's final
  runs; Proposal 1 can be screened on the existing cache first.
- **Foil:** whatever method becomes the paper's fix must beat `zcore` (plan Appendix A).

## Suggested order

1. Jaccard check: `patchmean` vs `cls` subsets (minutes, no training).
2. Fix the `kmeans_hist` fitting (shared patch sample for Proposals 1 and 2, and Idea A).
3. Build the screening harness; screen Idea A2/A3 and Idea B2 (no training).
4. Proposal 1 screen at 0.2/0.3 × 3 seeds, in parallel with the `patch_mean` batch.
5. Proposal 2 (with aspect-preserving features), compared against `zcore`.
6. Proposal 3 (motivated by Idea B4) if 1 and 2 only partly recover the street-furniture
   classes.
