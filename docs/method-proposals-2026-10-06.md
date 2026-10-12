# Method proposals after the prototypicality batch (2026-10-06)

Three candidate methods to run next, alongside the planned `patch_mean` prototypicality batch
(3 k × 5 ratios × 3 seeds = 45 runs). Written after Experiments 1–7 and the 2026-10-06
analysis in `docs/results_wandb.md`. Nothing here has been implemented or run yet.

## Reassessment after the `patch_mean` batch (2026-10-10)

**This section supersedes the "Suggested order" and the S/T prioritization table further
down.** *Update 2026-10-11: the literature check
(`docs/literature-survey-proposals-2026-10-11.md`) adds **FreeSel** as a must-have baseline
next to `zcore` (label-free, one-shot, dense features, already run on Cityscapes
segmentation). It also demotes Proposal 3 further: region-vs-image gains are already
published.* It's based on Experiments 1–9 and the 2026-10-10 analysis in `docs/results_wandb.md`.
The older sections are kept as the record of each idea.

### What changed

1. **The "patch mean ≈ CLS" prediction failed, in the useful direction.** Keep-hard
   prototypicality on patch-mean embeddings beats random at 0.1–0.3 for every k, and beats
   CLS by about +1 mIoU at 0.2–0.3 and +2.75 at 0.1. Its subsets overlap little with CLS's
   (Jaccard 0.29–0.47). The Jaccard check (old step 1) is done.
2. **Two trends that haven't flattened yet:**
   - More local is better: CLS < patch mean.
   - Finer prototypes are better for patch mean: k10 < k20 < k50 at every ratio ≤ 0.3. CLS has
     no such trend.
   Both point toward measuring atypicality more locally. This is the strongest evidence so far
   for Proposal 1 and for mechanism (c).
3. **The gain grows as the subset shrinks**, up to +3.0 mIoU and +11.6 rare mIoU at 0.1 for k50.
   The old advice to skip 0.1 in screens is wrong for patch methods: 0.1 is where they
   separate most.
4. **The rare-class gain is narrow.** Most of it is `train`. Without `train` it is about +3.7
   at 0.2 and not significant at the other ratios. Traffic light, pole and sign still lose,
   about half as much as with CLS. **Street-furniture recovery is still the open problem**,
   and none of the image-level scores tried so far fixes it.
5. **The paper's Contribution 1 as written ("label-free selection fails to beat random for
   segmentation") does not match Cityscapes.** The data supports a different headline:
   *the representation decides whether label-free selection helps, and global selection
   creates a structured per-class trade-off*. Contributions 2 and 3 get stronger, but the next
   experiments must then answer three questions before any new method:
   (i) is the win robust? (ii) does dense beat the strongest *global* method, `zcore`?
   (iii) *why* does it win?

### Validity problems in the current numbers

- **Boundary F isn't comparable between random and prototypicality.** Commit `12cc5ed`
  changed boundary F to exclude ignore-adjacent pixels. All `cs_random` runs predate it; every
  prototypicality run uses the new metric. That's why prototypicality at 0.7 "beats" full data
  on boundary F (50.5 vs 45.1). mIoU and rare mIoU are unaffected. Fix: re-score the random
  checkpoints (`results/checkpoints/cs_random_*` exist) with `04_eval.py`. No training needed.
- **The random band rests on 3 seeds.** A std from n = 3 is crude (random's std is 0.98 at 0.5
  but 0.52 at 0.3), and every ▲/▼ verdict depends on it. Prototypicality subsets barely change
  with the seed, so the method's std is mostly training noise. Two more random seeds at
  0.1/0.2/0.3 is the cheapest way to make all verdicts trustworthy.
- **Unknown attribution.** `bal50` spends half the budget spreading picks evenly across
  clusters. The gain could come from *keeping atypical images* or from *covering scene
  clusters evenly*. That decides whether to develop Proposal 1 (atypicality) or Proposal 2 /
  `zcore`-style coverage, so test it before building either.
- **The patch cache squashes images 4× horizontally** (audit #6), and the classes that still
  lose (traffic light, pole, sign) are exactly the thin vertical ones. Now that patch features
  are the winning representation, every new patch method should be built on aspect-preserving
  features. Otherwise each one gets run twice.

### New priority table

Cost uses the measured per-run times: one 0.1/0.2/0.3 × 3-seed block ≈ 3 GPU-h.

| Tier | # | What | Cost | Why now (what result motivates it) | What it decides |
|---|---|---|---|---|---|
| **0: validity** (do first, mostly no training) | 0a | Re-score `cs_random` checkpoints with the current eval | ~2 h eval, no training | Boundary F is computed two different ways (see above) | Whether boundary F can be reported at all |
| | 0b | Random seeds 3, 4 at 0.1/0.2/0.3 | 6 runs, ~2 GPU-h | Band from n = 3; every verdict depends on it | Trustworthy ▲/▼ for all existing and future rows |
| | 0c | Composition analysis of the 7 existing subset families (harness step 2): per-class image/pixel counts vs random, plus per-class IoU table and image-rare metric (audit #5) | hours, CPU | Gains are narrow (`train`) and losses specific (traffic light, pole) | Mechanism evidence for Contribution 2; checks whether composition predicts per-class IoU |
| **1: decisive, cheap** | 1a | **Attribution controls** on patch-mean k50: `prototypicality-easy` (preset exists) and cluster-stratified random (same clusters, random within cluster; small code change) | 18 runs, ~6 GPU-h | We don't know whether atypicality or cluster coverage drives the gain | **The fork:** atypicality wins → Proposal 1 first; stratified random ≈ hard → coverage (Proposal 2) first and the "hard" story is wrong |
| | 1b | **`zcore`** on CLS (and on patch mean if cheap) | ~1 day code + 9 runs, ~3 GPU-h | Plan §5.1a says it's non-optional. It matters more now: since selection *does* beat random, the claim becomes "dense beats the best global method" | Whether "dense > global" survives the strongest global foil. If `zcore`-CLS ≈ patch-mean proto, the representation claim is in trouble |
| | 1c | **Aspect-preserving re-extraction** (518×1036) + rerun patch-mean k50 | ~1 GPU-h extraction + 9 runs, ~3 GPU-h | Losing classes are thin and vertical; the squash hits them hardest | A cheap test of mechanism (c) on its own. Every patch method below is built on this cache |
| | 1d | Fix the `kmeans_hist` fitting (shared patch sample, audit #4), on the new cache | code only | Prerequisite for 2a, 2b, 3a and the label-free proxy | — |
| **2: the method** | 2a | **Proposal 1 (patch tail) + Idea A2 (IDF-weighted pooling)** as the next points on the locality axis: CLS → mean → IDF-weighted → top-q tail, at 0.1/0.2/0.3 | ~2 blocks, ~6 GPU-h | Both trends (local > global, larger k better) predict it improves further | A dose-response figure for the mechanism. Does more locality recover traffic light/pole? |
| | 2b | **Proposal 2 (rarity-weighted patch coverage)**, hybrid with *patch-mean* atypicality (not CLS as originally written) | ~1–2 days code + ~2 blocks | The only idea that explicitly protects street furniture inside ordinary scenes, which atypicality scores keep dropping | Contribution 4. Decisive rows: vs `zcore` and vs patch-mean proto. Order 2a/2b by the result of 1a |
| | 2c | **Second dataset** (ADE20K per the plan, CamVid as a cheap pilot): calibrate the LR, then random + patch-mean k50 + `zcore` at 0.1/0.2/0.3 | LR sweep + ~3 blocks (ADE20K ≈ 7× Cityscapes per epoch) | All evidence is from one dataset with very uniform scenes, which may favor atypicality selection | Whether the headline generalizes. Start the LR calibration in parallel with Tier 1 |
| **3: conditional** | 3a | S5 quality-weighted DPP | medium | A sibling of Proposal 2 (both submodular quality + diversity). Nothing in the results calls for it specifically | Only if 2b works (DPP as a head-to-head) or fails in a way DPP's spread-in-feature-space would fix |
| | 3b | Patch-mean k = 100/200 | 6–9 runs, no code | The k trend hasn't flattened | Run when the GPU would otherwise idle. Also informs K for 2a |
| | 3c | S1/S2 patch rarity scores in the harness | low, no training | Alternative axis-1 scores if prototype distance turns out noisy in 2a | Train only if they beat prototype distance on the label-free proxy |
| | 3d | Proposal 3 (region-level, labeled-pixel budget) | ~1 day trainer work + new framing | Use it if 2a/2b still can't recover traffic light/pole: that would show image-level selection is the limit (mechanism b) | A second paper section or follow-up |
| **Backlog** | — | B2 jigsaw sensitivity, S3 effective rank (analysis only: does patch-mean-hard just pick cluttered scenes?), S4, T1–T5 | — | Nothing so far suggests the patch vocabulary is the bottleneck, which is the problem T1/T4 solve | Revisit after 2b |
| **Drop** | — | B1, B3, A5, A6 | — | Already argued against above | — |

### Recommended next steps

1. **Today (no GPU contention):** 0a and 0c, and start 1b (`zcore`) and 1c (re-extraction)
   coding.
2. **Next GPU slots, in this order:** 0b → 1a → 1c's patch-mean rerun → 1b. That's about
   14 GPU-h total, and every result feeds a decision.
3. **Then** 2a or 2b first, depending on 1a. Start 2c's LR calibration in parallel.

**Screening protocol change:** screen at **0.1/0.2/0.3 × 3 seeds**. Drop 0.5/0.7 from screens:
random's band there is too wide (0.98 at 0.5), and every method gains less than 1 point there.
Keep the full 5-ratio ladder for final rows only.

**Why `zcore` and the attribution controls come before any new method:** the results have
already moved the paper from "nothing beats random" to "dense label-free selection works".
That claim has two obvious holes a reviewer would find: a stronger global baseline, and an
explanation of *what* about the selection helps. A new method built before those are closed
could end up being explained away by either one.

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

## Further proposals under debate (added 2026-10-10)

Two more families of ideas. **Family S** uses linear algebra on the frozen DINOv2 patch
features (no training). **Family T** trains a small label-free model on the frozen features
to reshape the space before selecting. Nothing here is implemented.

They're numbered **S1–S5** and **T1–T5** so they don't clash with Idea A (A1–A6) and Idea B
(B1–B4) above. "Assessment" notes are review comments added when writing them up. Novelty
claims haven't been checked against the literature yet.

### How these ideas fit together

Most of the ideas in this document are choices on three independent axes:

| Axis | Options so far |
|---|---|
| **1. Patch rarity score** (how unusual is this patch?) | distance to nearest patch prototype (Proposal 1), concept IDF (Idea A2), Mahalanobis distance (S1), residual after removing common directions (S2), flow log-likelihood / AE error (T2), masked-prediction surprise (T3) |
| **2. Aggregation to an image** | mean (`patch_mean`), weighted mean (Idea A), top-q tail (Proposal 1), softmax with temperature (Idea A3), image-level structure scores (S3, S4) |
| **3. Selector** | rank and keep top (prototypicality), weighted coverage (Proposal 2), quality-weighted DPP (S5), regions instead of images (Proposal 3) |

Presenting the method as one point in this grid, with ablations along each axis, is cleaner
for the paper than a list of separate methods. It also keeps the experiment count down:
screen axis 1 with the harness first, then only combine the winners with axes 2 and 3.

---

### Family S: Spectral / SVD ideas (no training)

#### S1: Inverse-spectral reweighting ("whitening as a learned IDF")

**In plain terms.** Most patches are road, building and sky, so most of the variation in the
patch features lies along a few "common" directions. Rescale the feature space so that rare
directions count as much as common ones. Then a patch that differs from the norm along a
direction few patches use stands out as far away.

**How.**
1. Stack all patches into a matrix X (patches × 768) and center it.
2. SVD: X = UΣVᵀ.
3. Re-embed: X′ = X · V · diag(σ⁻ᵅ).
   - α = 1: full PCA whitening.
   - 0 < α < 1: a tunable tilt between the original space (α = 0) and full whitening.

**Why it's principled.** Euclidean distance in the whitened space equals Mahalanobis distance
in the original space. Mahalanobis distance is, up to scale and a constant, the negative
log-likelihood under a Gaussian fitted to the patches. So "distance from the mean" becomes a
principled rarity score, and any k-center or coverage selector run in X′ treats outliers
along low-variance directions (candidate rare regions) as far away and worth keeping. It's
the continuous, geometry-based version of the IDF idea (Idea A2), with no clustering step.

**Caveat (also the theory).** Full σ⁻¹ blows up the smallest singular directions, which are
often noise. Use shrinkage or a regularized covariance (e.g. Ledoit–Wolf), or α ≈ 0.5.
Check that the amplified directions are semantic by visualizing the patches that light up.

**Nearest prior art.** Mahalanobis scoring for out-of-distribution detection. The novelty
would be using it as the selection metric for a dense coreset.

**Assessment.**
- Cheap: the 768×768 covariance can be accumulated in one streaming pass over the cache
  (about 4M patches); no need to hold all patches in memory.
- The cached patches are L2-normalized and come from squashed images (audit #6). Both
  change the spectrum, so fix the aspect ratio before trusting the low-variance directions.
- S1 is a patch rarity score (axis 1). It still needs an aggregator (axis 2) and a selector
  (axis 3); S1 + top-q tail + coverage/DPP is the natural combination.

#### S2: Dominant-subspace nullout ("rarity as residual")

**In plain terms.** Find the few directions that explain common content, delete them, and
see what's left. A patch with a lot left over isn't explained by common content, so it's a
candidate rare object or boundary.

**How.**
1. Estimate the top-k principal directions from frequent patches (or all patches).
2. Project every patch onto the complement of that subspace.
3. Score each patch by its residual norm.

**Why.** It's the explicit, hard-threshold version of what S1 does softly. It also matches
the known DINOv2 behavior that the first PCA components separate background from
foreground.

**Assessment.**
- Very cheap, one knob (k), and gives a per-patch heatmap that's easy to check visually.
- A good first candidate for the screening harness, alongside Proposal 1's prototype
  distance. If S1 and S2 rank patches similarly, keep only the simpler one (S2).
- Like S1, it may highlight boundaries as much as rare objects. That might still help
  segmentation (boundaries are hard), but it's a different claim; the heatmaps will show
  which one it is.

#### S3: Effective rank / spectral entropy as image informativeness

**In plain terms.** Count how many genuinely different "things" are in an image. A frame of
mostly sky and road has few; a busy street with many object types has many. Rare classes
almost never appear in near-empty frames.

**How.** For each image, take the singular values of its own 1,369 × 768 patch matrix,
normalize them to sum to 1 (σ̃), and compute the effective rank `erank = exp(H(σ̃))`, where H
is the entropy. Then bias selection toward high-erank images, or cover the erank
distribution.

**Why.** Effective rank is a standard, label-free measure of how many independent patterns a
matrix contains, so it's a theoretically grounded proxy for "compositionally rich, relevant
for dense segmentation".

**Assessment.**
- Practically free: 2,975 small SVDs.
- Risk: it measures *clutter*, not *rarity*. A busy downtown scene full of common objects
  scores high. It may also overlap heavily with keep-hard CLS prototypicality (busy scenes
  are atypical). Check Jaccard against the CLS subsets first.
- Probably best as a secondary term (e.g. the quality score q in S5), not a selector alone.
- Decide whether to center each image's patch matrix first; it changes what the top
  singular value means.

#### S4: Within-image Laplacian eigenvectors ("free segments")

**In plain terms.** Split each image into unsupervised segments using only patch similarity,
then use the segments as selection signals: how many segments an image has, and whether it
contains a small, isolated segment (a likely small object).

**How.** Per image:
1. Build a patch-affinity graph (cosine similarity between patches).
2. Form the normalized graph Laplacian and take its smallest non-trivial eigenvectors.
   This is normalized cuts; the eigenvectors are soft segmentations.

Two signals follow:
- **Number of eigen-segments:** a structure/complexity score, like S3 but spatial.
- **A small, isolated eigen-cluster:** a tiny coherent region, likely a small or rare object.

**Prior art.** Deep Spectral Methods (Melas-Kyriazi et al., CVPR 2022) show this gives
genuine unsupervised segments from self-supervised ViT features.

**Assessment.**
- Moderate cost: one 1,369 × 1,369 eigenproblem per image, about 1 s each.
- On the current 37×37 grid from squashed images, a traffic light is roughly one patch, too
  small to form its own segment. This idea needs the aspect-preserving (37×74) features from
  audit #6 to have a chance.
- The "small isolated segment" signal is the part that targets rare objects; the segment
  count largely duplicates S3.
- (The pasted notes call this the rigorous version of a "scene-structure descriptor (M5)"
  from an earlier discussion; M5 isn't recorded in this document.)

#### S5: Quality-weighted DPP (the strongest candidate in this family)

**In plain terms.** Pick a subset that is both *high quality* (contains rare things) and
*diverse* (no two picks are near-duplicates), using a single objective with a mathematical
guarantee.

**How.** A determinantal point process (DPP) picks a subset S with probability ∝ det(L_S).
The determinant is the squared volume spanned by the selected items' feature vectors, so it
is large only when the items point in different directions (diverse).

The lever is the **quality–diversity decomposition**:

```
L_ij = q_i · S_ij · q_j          →          det(L_S) = (Π_{i∈S} q_i²) · det(S_S)
```

- q_i = rarity/quality score of image i (from S1/S2, T2, or Proposal 1's tail score).
- S_ij = similarity between images i and j.
- The determinant multiplies "pick rare things" (the q product) against "pick a spanning,
  non-redundant set" (det(S_S)).
- log det is submodular, so greedy MAP selection has a (1 − 1/e) guarantee.

**What would make it novel for this problem.**
1. **Build S between patch sets, not CLS vectors**: a kernel between two images' patch-token
   distributions (a mean-map / MMD kernel, or an optimal-transport kernel), so diversity is
   measured densely.
2. **Set q from rare-patch density**, so the tail gets pulled in.

**Nearest prior art.** DPP-based coresets exist for classification. A rarity-weighted,
patch-distribution-kernel DPP for label-free dense segmentation appears unclaimed, **but this
has not been checked yet**.

**Assessment.**
- **Linear-kernel trap:** an MMD/mean-map kernel with a *linear* base kernel reduces exactly
  to the dot product of patch means, i.e. `patch_mean` similarity. The kernel between patch
  sets must be nonlinear (e.g. RBF mean map via random Fourier features, histogram
  intersection over fixed patch concepts, or OT between per-image concept centroids) to add
  anything.
- **Cost:** exact MMD between all image pairs is about 4.4M pairs × (1,369 × 1,369) patch
  comparisons, too expensive. Use sampled patches, random features or concept histograms.
  Greedy DPP itself is cheap (fast incremental-Cholesky greedy, about N·B² operations; under
  a minute even at ratio 0.7).
- **Relation to Proposal 2:** DPP and weighted coverage are both submodular quality +
  diversity objectives. Coverage rewards *covering each concept enough*; DPP rewards
  *spreading out in feature space*. They should be compared head-to-head, and both against
  `zcore`.
- q needs a temperature/exponent, since det(L_S) scales with q²; too strong and it becomes
  "rank by q", too weak and it becomes plain diversity. Set it with the label-free proxy.

---

### Family T: Trained post-processing on the embeddings (still label-free)

Same spirit, but a small model trained on frozen DINOv2 features reshapes the space before
selection. All are label-free; T5's stretch version is not.

#### T1: Projection head with a spatial-contrastive objective (STEGO-style)

**In plain terms.** Train a tiny network on top of the patch features so that patches
belonging to the same region end up closer together and different regions further apart.
Clustering afterwards gives cleaner pseudo-classes, including tighter clusters for rare
stuff.

**How.** Train a small MLP on patch tokens: pull together patches that are both spatially
adjacent and feature-similar, push the rest apart.

**Prior art.** STEGO (Hamilton et al., ICLR 2022) distills DINO feature correspondences into
a segmentation-friendly projection, fully unsupervised, and works well for segmentation. The
novelty would be using a STEGO-style projector as a refinement *before selection* rather
than as the segmenter.

**Why.** It amplifies the correlation structure dense tasks rely on and that CLS-style
invariance throws away.

**Assessment.**
- Improves the *vocabulary* (axis 1 input) rather than being a selector, so it pays off only
  through Proposal 2, S5 or Idea A. Measure its effect with the harness's rare-concept proxy.
- STEGO's own training recipe has several sensitive hyperparameters; budget time for that.
- A reviewer may ask whether gains come from the projector or from STEGO-level
  segmentation quality leaking in; an ablation with and without the projector covers it.

#### T2: Density model for calibrated rarity (flow or autoencoder)

**In plain terms.** Learn how likely each kind of patch is, then call low-likelihood patches
rare. An image's score comes from its rarest patches.

**How.** Fit a normalizing flow (exact log-likelihood) or, cheaper, an autoencoder
(reconstruction error as a proxy) on the patch embeddings. Aggregate each image's top-k
rarest patches (top-k, never mean, to avoid washing out the signal).

**Why.** Proper density estimation instead of approximating it with distances, which
concentrate in high dimensions; the coreset-difficulty literature keeps running into that
weakness.

**Assessment.**
- Caveat: deep density models are known to assign likelihood by low-level statistics rather
  than semantics (Nalisnick et al., 2019, "Do deep generative models know what they don't
  know?"). Glare, blur or the ego-car hood could look "rare". Check with heatmaps.
- S1 is the Gaussian special case of T2 at almost no cost. Run S1 first; T2 is only worth it
  if S1's Gaussian assumption is clearly the bottleneck.

#### T3: Masked-patch surprise (reuse the masked-prediction signal)

**In plain terms.** Hide a patch and ask how well it can be predicted from its neighbors. If
it's hard to predict, it's locally surprising: a boundary or rare content.

**How.** Train a small model to predict a patch token from its neighbors, or measure it with
the frozen model by masking. Score = prediction residual.

**Why.** Masked-prediction residual estimates local information content directly. It's a
label-free stand-in for the prediction uncertainty that region-based active learning uses,
a signal normally unavailable before annotation.

**Assessment.**
- DINOv2 was trained with an iBOT masked objective, but the torch.hub weights ship only the
  backbone, not the iBOT head. The frozen-model version would compare a patch's token with
  vs. without masking (the backbone accepts a mask token) rather than reuse the head.
- Masking each of 1,369 positions per image separately is expensive; mask random blocks
  (e.g. 25% per forward pass, a few passes per image) and average.
- Requires re-running DINOv2 (GPU), so it would compete with training runs for the GPU.

#### T4: Pseudo-label metric-learning purifier (self-training loop)

**In plain terms.** Cluster patches into pseudo-classes, train a small projector so patches
of the same pseudo-class move closer, re-cluster, and repeat once or twice. Each round makes
the clusters, including the rare ones, tighter and better separated.

**Why.** Even noisy pseudo-labels usually improve cluster separability; that's why
self-training works.

**Risk.** Confirmation of pseudo-label errors: a rare object merged into a common cluster in
round 1 gets pulled further into it. Cap the iterations.

**Assessment.**
- Depends on a good first clustering, i.e. the `kmeans_hist` fix (audit #4).
- The pasted proposal says to "sanity-check tail clusters against ground truth once". That
  is fine as an analysis, but it must not be used to tune the loop (number of rounds, K),
  or labels leak into the method. Choose those with the label-free proxy and report the
  ground-truth check separately.
- Overlaps with T1 (both refine the vocabulary). Pick one; T1 has the stronger prior art.

#### T5: (Frontier) Information-bottleneck projector / learned selector

**In plain terms.** Train a projector that keeps exactly the information that separates
regions and drops the rest, the theoretically ideal feature for selection. Further out, learn
a function that predicts how much each image would add to rare-class mIoU.

**How.**
- IB projector: an information-bottleneck objective with spatial adjacency as the
  "relevance" variable.
- Learned selector: meta-learn an image's marginal contribution to rare-class mIoU.

**Assessment.**
- The IB projector is a research project of its own; estimating mutual information in
  768 dimensions is hard and unstable.
- The learned selector needs labeled training runs, which **breaks the label-free
  constraint**. Keep it as a stretch chapter or future work, not the core.

---

### Prioritization for S and T

*Superseded by the 2026-10-10 reassessment at the top of this document.*

| Idea | Cost | Novelty | Expected payoff | Suggested status |
|---|---|---|---|---|
| S2 residual | very low | low–medium | medium | **screen first** (harness, axis 1) |
| S1 whitening | low | medium | medium | **screen first**; drop if it ranks like S2 |
| S5 quality-weighted DPP | medium | **high** | **high** | **main candidate**, with q from the best axis-1 score; compare against Proposal 2 and `zcore` |
| S3 effective rank | very low | low | low–medium | secondary term (q in S5), after a Jaccard check |
| S4 Laplacian segments | medium | medium | medium | only after aspect-preserving features (audit #6) |
| T3 masked surprise | medium (GPU) | medium–high | medium | later; needs GPU time |
| T1 STEGO projector | medium–high | medium | medium | later; only if the vocabulary is the bottleneck |
| T2 flow / AE | medium | low–medium | low–medium | only if S1 is clearly limited by its Gaussian assumption |
| T4 self-training purifier | medium | low–medium | low–medium | pick T1 instead |
| T5 IB / learned selector | high | high | uncertain | stretch / future work |

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

## Suggested order (original, superseded)

*Superseded by the 2026-10-10 reassessment at the top of this document. Step 1 is done
(Jaccard 0.29–0.47).*

1. Jaccard check: `patchmean` vs `cls` subsets (minutes, no training).
2. Fix the `kmeans_hist` fitting (shared patch sample for Proposals 1 and 2, and Idea A).
3. Build the screening harness; screen Idea A2/A3, Idea B2 and the patch rarity scores
   S1/S2 (no training).
4. Proposal 1 screen at 0.2/0.3 × 3 seeds, in parallel with the `patch_mean` batch.
5. Proposal 2 and S5 (quality-weighted DPP), with aspect-preserving features, compared
   head-to-head and against `zcore`.
6. Proposal 3 (motivated by Idea B4) if 1 and 2 only partly recover the street-furniture
   classes.
