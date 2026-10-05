# Experiment & System Design Plan
## Label-Free Data Selection for Semantic Segmentation (Workshop-Paper Target)

**Owner:** (you)
**Hardware:** 1× NVIDIA RTX 3090, 24 GB VRAM
**Target venue:** CVPR/ICCV/ECCV/NeurIPS **workshop** (data-centric AI / learning-with-limited-data / efficient-CV family). Main-conference upside if the diagnostic result is strong.
**Timeline:** ~3–4 weeks of part-time running. No single run exceeds ~2.5 h.

This document is the single source of truth for: the paper narrative, datasets, models, the fixed training recipe, baselines, selection methods, metrics, the experiment ladder with decision gates, hyperparameter policy, compute budget, and the software system to build (repo layout, configs, logging schema, CLI). It is written to be directly usable as a build spec for Claude Code.

---

## 0. Paper narrative (decide this first — everything serves it)

**Framing: the paper is a DIAGNOSTIC/CHARACTERIZATION study. A new method is an optional bonus, not the spine.** This is the single most important design decision in the plan: it means the paper is *complete and publishable even if no selection method ever beats random*, because the contribution is the finding and its explanation — which is in your control — not a method win, which is not.

**The spine (these three exist regardless of whether a method works):**
- **Contribution 1 — the diagnostic (carries the paper):** label-free selection methods that reliably beat random for *classification* (prototypicality, global-embedding coverage, ZCore) **fail to beat random for semantic segmentation**, and the failure is *structured* — it concentrates in **rare classes**. Shown with matched compute + seed variance bands on two standard benchmarks.
- **Contribution 2 — the mechanism (what makes it a real paper, not a "random wins" table):** *why* the classification playbook doesn't transfer to dense prediction. Candidate explanations to test and report (whichever the data supports): rare classes co-occur with common ones in the same images, so image-level selection cannot isolate or protect them; per-pixel supervision makes pixel-class balance matter more than image count; global embeddings average away the local regions that carry rare-class signal. **The mechanism is the deliverable.**
- **Contribution 3 — the representation study (cheap, standalone insight):** which representation (classification-supervised vs self-supervised vs segmentation-encoder vs dense foundation model) selects better segmentation coresets — the first controlled answer for segmentation.

**The optional payoff (only if the data supports it):**
- **Contribution 4 — a fix:** a rarity-weighted, patch-level **coverage** objective that recovers rare-class performance without labels. If it works, it upgrades the paper from "characterization" to "characterization + fix." If it doesn't, **the paper stands on Contributions 1–3 unchanged** and the method appears as an honest ablation ("even an explicitly rare-class-aware coverage objective does not reliably beat random, which further supports the mechanism in Contribution 2").

**Framing hook (intro only, no extra experiments):** the same selector answers "what to label **first**, before any model exists" — the cold-start gap uncertainty-based segmentation active learning cannot fill.

**Closest prior work to distinguish from (cite explicitly, do not claim the space is empty):** *Core-Set Selection for Data-efficient Land Cover Segmentation* (Nogueira et al., arXiv 2505.01225, IEEE 2026) already does label-free + labeled coreset selection for segmentation with U-Net/SegFormer subset curves and significance tests — **in remote sensing**. Our distinctness: **natural images** (Cityscapes/ADE20K), a **rare-class-collapse diagnostic with a mechanism**, **dense/patch** representations, and comparison against the newest label-free coverage method (**ZCore, 2026**). The plain "select representative images for segmentation, beat random" paper is now published; ours must be the diagnostic.

**Golden rule:** the kill-shot (Stage 1) is a go/no-go **for the method, not for the paper**. Whatever it shows, Contributions 1–3 are your paper. Run it before building any method.

**What would make this NOT publishable (the one real risk):** a null result with no mechanism — i.e., "random is hard to beat" restated in a new domain. That finding alone is already known from classification (CCS). The paper lives or dies on the **segmentation-specific mechanism** in Contribution 2. Produce that, and the no-method outcome is a real paper; skip it, and it's a null result dressed up.

---

## 1. Scope and non-goals

**In scope:** one-shot, label-free (no ground-truth masks used during selection), image-level subset selection for training semantic segmentation; retention of mIoU at matched compute; rare-class behavior; cross-architecture generalization.

**Explicit non-goals (state these in the paper to pre-empt reviewers):**
- Not chasing published SOTA mIoU. We use a short, fixed *proxy* recipe; absolute numbers are below full-schedule SOTA **by design**.
- No Mask2Former / COCO-scale training (out of hardware scope).
- Not instance segmentation (that's TFDP's turf and needs masks).
- Not dataset distillation/condensation.
- Warm-start uncertainty active learning is a *reference*, not our track.

---

## 2. Datasets (locked)

| Role | Dataset | Train / Val | Classes | Why | Notes |
|---|---|---|---|---|---|
| Pilot / debug | **CamVid** | 367 / 101 (+233 test) | 11 | tiny, fast plumbing shakeout | 720×960 → train at 480×480 crops |
| **Main testbed** | **Cityscapes (fine)** | 2975 / 500 | 19 (eval) | strong long-tail → the rare-class thesis lives here | 1024×2048; train at 512×1024 crops |
| Scale/diversity | **ADE20K** | 20210 / 2000 | 150 | scalability + many-class diversity; matches InfoBatch setting | run **selectively** (method vs random vs best baseline only) |
| Optional swap | Pascal VOC aug | 10582 / 1449 | 21 | alt "fast + many images" | only if CamVid feels too small |

**Rare-class definition (Cityscapes, precompute once from pixel frequency):** the ~5 rarest of the 19 eval classes — typically `wall, fence, pole?, rider, truck, bus, train, motorcycle`. Compute exact pixel-frequency ranking from the **full** train set labels **once** (labels are allowed for *evaluation/analysis*, just not for *selection*), freeze the bottom-K list, and report rare-class mIoU against it.

**Preprocessing (fixed, identical across all runs):** standard mean/std normalization; training augmentation = random scale [0.5, 2.0], random crop to the per-dataset crop size, random horizontal flip, (Cityscapes/ADE) photometric jitter. Val = single-scale, whole-image (sliding-window for Cityscapes at 512×1024 stride).

---

## 3. Models

**Primary workhorse — everything except Stage 4:** **SegFormer-B0 (MiT-B0)**, ImageNet-1k-pretrained encoder (HuggingFace `nvidia/mit-b0`). ~3.7 M params, transformer, fast, strong, standard. This is the model for the diagnostic, representation study, and method development. For a *selection* paper the model is instrumentation, not contribution — it must be fast, credible, and standard, and B0 is all three. Do **not** upgrade to B2+ (slower every run, eats seed budget) or Mask2Former (too heavy for the seed budget on one 3090, and its mask-classification formulation muddies the story).

**Generalization set (Stage 4 only) — ONE CNN is sufficient:** **DeepLabV3+ (ResNet-50)** via `segmentation_models_pytorch`. It is the standard modern-CNN segmentation baseline and gives the cleanest "different architecture family" contrast. Keep **U-Net (ResNet-34)** only as a fallback if DeepLab's per-run time (~2–3× B0) bites. **One transformer (B0) + one CNN (DeepLabV3+) is the accepted bar** for a cross-architecture generalization claim at workshop and most main-conference selection papers; a third architecture adds cost without strengthening the claim.

**Why cross-architecture matters (state this explicitly in the paper):** the likely reviewer objection is *"your coreset was selected with DINOv2 features and trained on a transformer — you're just matching representation families."* Two things pre-empt it. (1) **Selection is already architecture-decoupled by design:** subsets are chosen with *frozen DINOv2* features, not with SegFormer's own features — so the training architecture never influences selection. Say this before the experiment even lands. (2) **Stage 4 proves transfer empirically:** select the subset once, then show it helps a network you did *not* select for (DeepLabV3+). Selection decoupling + one cross-architecture transfer result is the full defense.

**Multiple models — do you need them?** Yes, but only for the transfer defense above (one CNN in Stage 4), not for "more architectures = better." The diagnostic itself (Contributions 1–3) runs on B0 alone.

**Libraries:** HuggingFace `transformers` for SegFormer; `segmentation_models_pytorch` for DeepLabV3+/U-Net; `torch.hub` for DINOv2; `faiss` (or `scikit-learn` MiniBatchKMeans) for clustering; plain PyTorch training loop. **Do not** adopt mmsegmentation unless already fluent — setup overhead isn't worth it for a POC.

---

## 4. The fixed "proxy training recipe" (the core cost-control trick)

Every training run uses the **same** recipe; the **only** variable is which images are in the subset. This turns "train a segmentation model" (days) into "rank selection methods" (30–90 min/run).

**Fix epochs, NOT iterations (ADR 0001 — see `docs/adr/0001-epoch-based-training.md` for the full rationale, including why this reverses an earlier version of this plan).** Every run trains for the *same number of epochs* over whatever subset it was given, so a 10% subset does ~10% of the full run's gradient steps — less total compute, not more repetition of the same images. This matches the closest directly-comparable prior work (Nogueira et al. 2505.01225, label-free coreset selection for segmentation), which trains every subset for a fixed epoch count and reports per-epoch time specifically to *quantify* the compute savings coreset selection buys — the efficiency story is a reported result here, not a confound to be held fixed. (We do not copy their literal epoch count — our datasets differ in size and hardware is one 3090 — see the ADR for how each dataset's epoch count was calibrated.) A **fixed-iteration** run (does the method ranking change under matched compute instead?) is kept as a secondary appendix ablation, not the primary convention.

| Setting | Cityscapes | CamVid | ADE20K |
|---|---|---|---|
| Crop | 512×1024 | 480×480 | 512×512 |
| Batch size | 8 | 16 | 8 |
| Epochs (fixed cap) | **100** = 37,100 steps on full data (160 for the one "ceiling" run) | 300 | 8 (16 for the one "ceiling" run) |
| Warmup | 6 epochs | 15 epochs | 1 epoch |
| Optimizer | AdamW | AdamW | AdamW |
| LR | **1e-4** (LR sweep on full data, see `cityscapes_proxy.yaml`) | 6e-5 (not yet calibrated) | 6e-5 (not yet calibrated) |
| LR schedule | poly (power 1.0, linear decay to 0) + linear warmup | same | same |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Precision | AMP fp16 | AMP fp16 | AMP fp16 |
| Est. wall-time/run (full data) | 1.27 h training (measured, 3 seeds) + ~0.3 h final eval | ~10–20 min | ~1.0–1.5 h |
| Est. VRAM (batch 8, crop 512×1024) | ~4.7 GB allocated / ~6.6 GB reserved (measured) | — | — |

Epoch counts are per-dataset (not a single number for every dataset) — ADE20K's full
dataset is ~7x Cityscapes', so the same epoch count would cost ~7x more compute there;
each dataset's count was calibrated to land close to the wall-clock this project's
hardware constraint (one 3090, no run over ~2.5h) already implied under the old
iteration-based budgets. Cityscapes was later raised from 50 to 100 epochs and its LR
from 6e-5 to 1e-4 after measurement and an LR sweep (ADR 0001, Amendment 3); the
Cityscapes column above is the frozen recipe actually used for every reported run. Schedule is `poly` (power 1.0 — a straight line, same
complexity as `cosine`), matching the SegFormer/mmseg literature-standard fine-tuning
recipe (ADR 0001, amendment 2) rather than a "fancy" choice; `cosine`/`constant` remain
available as documented alternatives.

Wall-time now **scales down with the subset ratio** (a 10% subset finishes in roughly 10% of the full-data run's time) — this is the efficiency claim the paper reports, not a side effect to normalize away.

**Two reference denominators (run once each, needed for retention ratios):**
1. **Full-100% @ matched epoch budget** (the proxy recipe's epoch count): the fair denominator. `retention = mIoU(subset) / mIoU(full@N-epochs)`.
2. **Full-100% @ long budget** (Cityscapes: 160 epochs, ADE20K: 16 epochs — `*_ceiling.yaml`): the true ceiling, reported once for context only.

**Discriminativeness check (MANDATORY, day 1):** confirm full-100%@N-epochs clearly beats 10%-random@N-epochs. If they tie, the recipe is too short to discriminate → raise `epochs` and re-check before running anything else. Also sanity-check the *other* failure mode this convention introduces: at the smallest ratio (5%), confirm the run still gets enough absolute gradient steps to clear warmup and reach a non-degenerate mIoU — if not, that ratio may need its own epoch floor (see the ADR's open question).

---

## 5. Baselines and selection methods

All selection happens **before** any training, using cached features. Selection is cheap (minutes). Group by track.

### 5.0 How a baseline earns its slot (the framework — reuse this every time a reviewer asks "why not X?")

A baseline is included because it satisfies **all four** criteria, not because it is recent:

1. **Distinct mechanism.** It occupies a point in the design space our method claims to improve on. Two methods that both "cluster global embeddings and pick representatives" are **one** baseline, regardless of publication year.
2. **In-track (label-free, one-shot).** Our claim is *no masks used during selection*. Methods needing labels, gradients, or a trained model are disqualified from the *competing* set; they may appear only as a fenced-off **labeled reference**.
3. **Strong + recent representative of its mechanism.** For each mechanism axis we want the strongest *current* version so no reviewer can dismiss it as a dated strawman. Recency is a **tiebreaker within an axis**, not an admission criterion on its own.
4. **Faithfully implementable on budget.** A botched reimplementation is worse than an omission — strawmanned baselines are punished harder than missing ones.

**Critical reframing:** we do **not** compare against other papers' reported numbers (those are classification results on CIFAR/ImageNet). We re-run each *selection mechanism* on our segmentation images and train our own fixed proxy segmenter on the result. So the question is never "is method M recent enough?" but "does M represent a design-space point I must cover, and is there a stronger/newer method representing the *same* point?"

**Mechanism axes we must cover (label-free, one-shot):** (a) random floor, (b) representativeness / prototypicality, (c) coverage / spread, (d) redundancy removal, (e) intrinsic per-image complexity. Our method is a **new point on the coverage axis** (dense/patch + rarity weighting), so the coverage axis must carry the strongest available foil.

### 5.1 Label-free selection track (these compete)

| ID | Mechanism axis | Representation | Recency role | Priority | Implementation notes |
|---|---|---|---|---|---|
| `random` | floor | none | — | must | 3–5 seeds; the variance band everything is judged against |
| `prototypicality` | representativeness | DINOv2 CLS | **canonical** (SSP, 2022) | must | dist-to-nearest-centroid ranking; try keep-hard & keep-easy. *Named mechanism representative — expected by reviewers; not a recency claim.* |
| `kcenter` | coverage (global) | DINOv2 CLS | canonical (2018) | must | greedy farthest-point min-max; O(n·B) |
| `semdedup` | redundancy | DINOv2 CLS | canonical (2023) | must | cluster, drop near-duplicates above cosine τ, random from remainder |
| `bpp` | intrinsic score | none (JPEG) | recent but weak (2024) | keep | rank by compressed bits/pixel; cheap label-free scalar to beat |
| **`zcore`** | **coverage (global, FM subspace)** | DINOv2 CLS / CLIP | **recent strong (2026)** | **must-add** | **the direct foil** — see §5.1a |
| `elfs` | representativeness (pseudo-label) | DINO + deep cluster | recent (2025), overlaps axis | optional | heavier: pseudo-labels + train-on-full-then-cutoff. Include only if clean code + spare time — see §5.1b |

Note: the earlier `kmeans_centers` entry is **folded into `prototypicality`** — with ZCore present they were the same representativeness/coverage point twice. Global k-means-centers now lives only as the internal "global" arm of our own method's ablation (§5.3), not as a standalone baseline.

#### 5.1a Why ZCore is non-optional (highest-value change to this plan)
ZCore (2026) is label-free, one-shot, builds a foundation-model embedding space, and selects by **coverage + redundancy reduction over that space**. That is precisely **the image-level, global-embedding version of our own method.** Our entire contribution reduces to the claim: *dense/patch + rarity-weighted coverage beats an otherwise-similar global coverage philosophy for segmentation.* Without ZCore (or a faithful reimplementation of its subspace-coverage mechanism), the central ablation "global coverage vs dense coverage" has a hole a 2026 reviewer will drive straight through. Its mechanism is simple enough to reimplement faithfully if the released code is unusable. **This is the fairest possible test of the paper's specific claim.**

#### 5.1b Why ELFS is optional, not a must-have
Two honest reasons it is *not* required: (1) its mechanism **overlaps an axis we already cover** — deep clustering on DINO embeddings sits next to prototypicality/representativeness; it is not a new design-space point. (2) It is a **pseudo-label-then-train-on-full-then-prune** pipeline, which is heavier and a correctness risk to reimplement on a deadline. ELFS is therefore a *recency box-check on an axis already covered*, not a distinct mechanism. Cite it; include it as a baseline only if clean public code exists and time is free; otherwise note in the paper that it is a heavier variant of the clustering/representativeness axis.

### 5.2 Reference / out-of-track (run for context, clearly fenced off — NOT part of the label-free claim)

The **label-free, one-shot constraint disqualifies the entire score/difficulty family** (EL2N, GraNd, Forgetting, InfoBatch, ELFS-cutoff), which needs labels, gradients, or a trained model. **State this exclusion as a scoping contribution**, not an omission: *"among truly label-free one-shot methods the field narrows to representativeness, coverage, redundancy, and intrinsic-complexity scoring; the score/difficulty family is out of scope by construction."* One sentence turns a perceived gap into a defined scope.

| ID | Method | Track | Why include |
|---|---|---|---|
| `full` | 100% data | reference | denominator + ceiling |
| `class_balanced` | label-aware balanced sampling | labeled reference | isolates "how much of the gap is just class balance?" |
| `infobatch` *(optional)* | dynamic loss-based pruning | labeled efficiency reference | the "how much can labeled dynamic pruning save on ADE20K" context number; fenced off from the label-free claim |
| `entropy_warm` *(optional)* | warm-model uncertainty | labeled/AL reference | connects to active-learning reviewers |

### 5.3 Your new method (Stage 3)

| ID | Method | Representation | Objective |
|---|---|---|---|
| `patch_coverage` | Rarity-weighted patch pseudo-class coverage | DINOv2 **patch** tokens | greedy submodular coverage over patch pseudo-classes, rare pseudo-classes up-weighted |

**`patch_coverage` spec:** (1) extract L2-normalized patch tokens for all images; (2) MiniBatchKMeans over *all patches* → K pseudo-classes (start K≈256); (3) each image → pseudo-class histogram `h_i`; (4) pseudo-class weight `w_k = 1/freq(k)`; (5) greedily maximize the monotone submodular coverage
`F(S) = Σ_k w_k · [1 − Π_{i∈S}(1 − h_i(k))]` s.t. `|S| ≤ B` (greedy → 1−1/e guarantee).
**Ablations:** rarity weighting on/off; patch vs global histograms; K ∈ {128, 256, 512}.

---

## 6. Representations to extract (cache once, reuse everywhere)

| ID | Model | Output used | Role |
|---|---|---|---|
| `dinov2_cls` | DINOv2 ViT-B/14 (`torch.hub`) | global CLS vector | global-embedding baselines |
| `dinov2_patch` | DINOv2 ViT-B/14 | patch token grid | your method + patch study |
| `rn50_sup` | ImageNet-supervised ResNet-50 | global pooled feat | classification-representation arm of Contribution 3 |
| `segb0_enc` | SegFormer-B0 encoder (ImageNet or short full-data trained) | pooled encoder feat | segmentation-representation arm |
| `clip_cls` *(optional)* | CLIP ViT-B/32 | global | extra representation point |

**ViT-B/14** (`configs/features/dinov2_vitb14.yaml`, 518×518 input, 768-dim) is the project default and is what every committed prototypicality subset uses; ViT-S/14 (the original choice) and ViT-L/14 remain available as configs. Extraction = one forward pass/image; cache to `.npy`/`.pt` keyed by image id. Full-corpus extraction: minutes (CamVid/Cityscapes), <1 h (ADE20K).

---

## 7. Metrics (compute all in one eval pass, log to CSV)

**Primary**
- **mIoU** (mean over eval classes), mean ± std across seeds, always plotted with random's variance band.

**The thesis metric**
- **Rare-class mIoU**: mean IoU over the frozen bottom-K classes. This is where the method should win if Contribution 1 holds.

**Secondary**
- Per-class IoU (full table, for the appendix).
- **Boundary F-score (BF)** / trimap-IoU at 3 px — segmentation-specific quality.
- Pixel accuracy.

**Efficiency / retention**
- `retention = mIoU(subset) / mIoU(full@matched-budget)`.
- **AUC-retention**: area under the retention-vs-ratio curve → one comparable number per method.
- GPU-hours per run, selection wall-time, total compute.
- mIoU-vs-GPU-hours plot (separate from the "which subset is more informative" question).

**Statistical rigor (reviewers will check):**
- Random: **3 seeds minimum, 5 for the final headline table.** Report mean ± std.
- Deterministic selectors: 2–3 **training** seeds (training init/order still varies results).
- A method "wins" only if its mean is outside random's ±1 std band; note where bands overlap. Do a simple paired comparison (e.g., Wilcoxon across seeds) for the headline claim.

---

## 8. Ratios and seed policy

- **Subset ratios:** {5, 10, 20, 30, 50} %. Core comparisons at **10/20/30**; add 5 and 50 to complete the curve once the method is chosen.
- **Seed policy:** a "seed" fixes numpy/torch/cuda RNG and, for stochastic selectors, the subset draw. Lean tier = 2 seeds on deterministic cells, 3 on random. Full tier = 3 everywhere, 5 for the final headline.

---

## 9. Hyperparameter policy (avoid combinatorial explosion)

**Two hard rules:**
1. **Tune the training recipe ONCE, on the FULL dataset only.** Small LR sweep `{3e-5, 6e-5, 1e-4}` + set the epoch cap by the discriminativeness check (§4, ADR 0001). LR schedule is `poly` (power 1.0, linear decay to 0) with linear warmup — the SegFormer/mmseg standard; `cosine`/`constant` exist only as documented fallbacks, no step/exponential variants (ADR 0001, Amendment 2). **Freeze it. Reuse identically for every subset and every method.** Never per-method tune the trainer — that's both unfair and budget-fatal.
2. **Selection-method hyperparameters set by cheap proxies, not by retraining.** k-means K, dedup threshold τ, prototypicality direction: pick by a **label-free proxy** (embedding-space coverage / silhouette / rare-pseudo-class recall) measured *without* training a segmenter. Only the final chosen setting gets trained. Report the proxy→final mapping so it's principled, not hand-tuned.

This keeps total training runs ≈ (methods × ratios × seeds) + references, with **no** hyperparameter grid multiplying it.

---

## 10. Experiment ladder (run in order; each gate can stop or pivot you)

### Stage 0 — CamVid pilot (½ day)
Purpose: shake out feature-extract → select → train → eval → log loop. Run `random`, `prototypicality`, `patch_coverage` at 10/20%. ~10 runs × ~20 min. Discard numbers scientifically; keep the code and the discriminativeness check.

### Stage 1 — Kill-shot on Cityscapes (the go/no-go)
Compare `random` (3 seeds), `prototypicality`, `kcenter`, `semdedup`, `bpp`, and **`zcore`** at 10/20/30% + full data @ the proxy budget (`cityscapes_proxy`, 100 epochs) + full data @ the ceiling budget (`cityscapes_ceiling`, 160 epochs). (`zcore` is the direct global-coverage foil — see §5.1a. It belongs in Stage 1 so the "global vs dense coverage" contrast is visible from the first result, not bolted on later.)
**Gate:** report overall mIoU **and rare-class mIoU**.
- If global-embedding selection **hurts rare-class IoU** vs random (even if overall ties) → Contribution 1 confirmed → proceed to full plan.
- If everything (incl. rare classes) sits inside random's band → **pivot**: drop the method, write the honest-benchmark + representation-study paper (Contribution 3 + reality-check), or move to the cold-start AL framing.

### Stage 2 — Representation study (Contribution 3)
Fix ratio = 20% and one selector (k-center or `patch_coverage`); vary only representation: `rn50_sup`, `dinov2_cls`, `dinov2_patch`, `segb0_enc`. 3 seeds. Answers "does dense/self-supervised representation select better segmentation coresets?" — standalone-publishable insight.

### Stage 3 — Your method + ablation (Contribution 2)
`patch_coverage` at 10/20/30% (3 seeds) vs Stage-1 baselines. **The decisive comparison is `patch_coverage` vs `zcore`** — same coverage philosophy, dense/patch+rarity vs global — because that isolates your actual contribution (§5.1a). Ablations: rarity on/off; patch vs global histograms (the "global" arm reuses the folded-in k-means-centers logic); K sweep. Target: recover the rare-class IoU that global selection lost.

### Stage 4 — Generalization (the strength claim / reviewer insurance)
Select once at 20% with `patch_coverage` and `random`; train **DeepLabV3+** and **U-Net** (2 seeds each). If the coreset helps across all 3 architectures, it's not representation-locked → major point.

### Stage 5 — ADE20K scale (optional, do if time)
`patch_coverage` vs `random` vs best baseline at 10/20% (2 seeds). Demonstrates the finding scales to many classes.

---

## 11. Compute budget

Stage 1 now carries **6 selection methods** (`random`, `prototypicality`, `kcenter`, `semdedup`, `bpp`, `zcore`); `zcore` adds ~9 runs (3 ratios × 3 seeds ≈ 11 GPU-h). Adding it is the single most defensible use of compute in the plan (§5.1a).

| Stage | Runs | ~h/run | Subtotal (GPU-h) |
|---|---|---|---|
| 0 CamVid pilot | ~10 | 0.4 | ~4 |
| 1 Kill-shot (Cityscapes, 6 methods) | ~45 | 1.2 | ~54 |
| 2 Representation study | ~12 | 1.2 | ~14 |
| 3 Method + ablation | ~24–30 | 1.2 | ~30–36 |
| 4 Generalization | ~8 | 2.3 | ~18 |
| 5 ADE20K (optional) | ~10–12 | 1.6 | ~16–19 |
| **Core (0–4)** | | | **~120–130** |
| **Lean core** (2 seeds on deterministic incl. `zcore`; drop 5%/50% first pass; defer `elfs`) | | | **~75–90** |

Run **lean core** first to get a submittable skeleton; backfill extra seeds and the 5%/50% ratios only where the story needs tightening. If budget is tight, cut `semdedup` or `bpp` (both weak/canonical) before cutting `zcore` — never cut the direct foil.

---

## 12. Software system to build (Claude Code spec)

### 12.1 Repo layout
```
segcoreset/
  configs/
    dataset/{camvid,cityscapes,ade20k}.yaml
    model/{segformer_b0,deeplabv3plus_r50,unet_r34}.yaml
    recipe/{cityscapes,ade20k,camvid}_proxy.yaml, {cityscapes,ade20k}_ceiling.yaml
                                   # frozen per-dataset, epoch-based recipes (ADR 0001)
    selection/{random,kmeans_centers,prototypicality,kcenter,
               semdedup,bpp,patch_coverage}.yaml
  segcoreset/
    data/            # dataset wrappers, transforms, rare-class freq
    features/        # dinov2/rn50/segb0/clip extractors + cache
    selection/       # one file per selector, common interface
    train/           # single fixed training loop (all models)
    eval/            # miou, rare_class_miou, boundary_f, retention
    logging/         # results store (CSV + optional SQLite)
    utils/           # seeding, config hash, git commit capture
  scripts/
    00_extract_features.py
    01_compute_rare_classes.py
    02_select.py
    03_train.py
    04_eval.py
    05_aggregate.py          # builds all paper tables
    06_plots.py              # retention curves, rare-class bars, miou-vs-gpuh
  results/
    runs.csv                 # append-only, one row per (method,ratio,seed)
    subsets/                 # cached selected id-lists (json)
    features/                # cached embeddings
  README.md
```

### 12.2 Common selector interface (so new ideas drop in trivially)
```python
def select(image_ids, features, budget, seed, cfg) -> list[str]:
    """Return the chosen subset of image_ids. No labels. Deterministic given seed."""
```
Every selector reads cached features and writes `results/subsets/{method}_{ratio}_{seed}.json`. Training reads that id-list. This decoupling means "try a new idea" = add one file, no trainer changes.

### 12.3 `runs.csv` schema (append-only; the whole paper is aggregated from this)
```
run_id, git_commit, config_hash, timestamp,
dataset, model, selection_method, representation,
ratio, seed, n_images, epochs, iterations,
miou, rare_class_miou, pixel_acc, boundary_f,
per_class_iou_json,
retention_vs_full_matched, gpu_hours_train, selection_seconds,
notes
```
Rules: one row per finished run; never overwrite; `05_aggregate.py` groups by (method, ratio) to produce mean±std tables; `06_plots.py` reads the same file. Provenance columns (`git_commit`, `config_hash`) are mandatory for reproducibility.

### 12.4 CLI flow (each stage is one command)
```
python scripts/00_extract_features.py --dataset cityscapes --repr dinov2_patch
python scripts/01_compute_rare_classes.py --dataset cityscapes --k 5
python scripts/02_select.py --dataset cityscapes --method kmeans_centers --ratio 0.2 --seed 0
python scripts/03_train.py  --subset results/subsets/kmeans_centers_0.2_0.json --recipe cityscapes_proxy --model segformer_b0
python scripts/04_eval.py   --run_id <id>            # appends metrics to runs.csv
python scripts/05_aggregate.py --dataset cityscapes  # -> paper tables (csv/latex)
python scripts/06_plots.py --dataset cityscapes      # -> figures
```

### 12.5 Reproducibility/engineering must-haves
- Global deterministic seeding (numpy, torch, cudnn) captured in every row.
- Log `git_commit` + a hash of the resolved config per run; refuse to run on a dirty tree unless `--allow-dirty`.
- Cache features once; never recompute.
- Evaluate at end-of-training only (plus one mid-checkpoint) — full Cityscapes val (500 imgs, sliding window) is not cheap across 60+ runs.
- One idempotent `runs.csv`; a `--dry-run` that prints the run matrix and estimated GPU-hours before launching.
- A `make killshot` / `make lean_core` target that enqueues an entire stage.

---

## 13. Deliverables per stage (what each produces for the paper)

| Stage | Table/Figure |
|---|---|
| 1 | **Fig 1 (headline):** rare-class IoU vs ratio — global-embedding selection (incl. `zcore`, the strongest global-coverage method) dips below random. **Table 1:** mIoU + rare-class mIoU, all baselines × ratios, mean±std. |
| 2 | **Table 2:** mIoU by representation (the transfer finding). |
| 3 | **Table 3:** method vs baselines, with **`patch_coverage` vs `zcore` as the headline row** (dense vs global coverage); **ablation table** (rarity on/off, patch/global, K). |
| 4 | **Table 4:** cross-architecture retention (SegFormer/DeepLab/U-Net). |
| all | **Fig 2:** retention curves + AUC-retention; **Fig 3:** mIoU-vs-GPU-hours. |

---

## 14b. The "no-method" paper — exactly how to position it and what to show

**Premise:** you finish all experiments and *no* selection method reliably beats random. The paper is still complete. Here is the clean positioning and the required evidence — nothing more, nothing less.

**Title framing (method-free):** *"When Does Label-Free Data Selection Help Semantic Segmentation? A Controlled Study."* The claim is a *question answered*, not a method sold.

**The four things you MUST show (all come from experiments already in this plan):**
1. **The non-transfer result (Contribution 1).** A clean table + figure: methods that beat random for classification (prototypicality, k-center, ZCore, SemDeDup) do **not** beat random for segmentation, on Cityscapes **and** ADE20K, at matched compute, with seed variance bands. Random's band overlapping every method *is the headline* — but only if it's rigorous (≥3 seeds, matched epochs).
2. **The mechanism (Contribution 2 — the load-bearing part).** An *explanation*, tested, for why classification selection fails on dense prediction. At least one of: (a) **rare-class co-occurrence** — measure how often rare classes appear only alongside common ones, showing image-level selection can't isolate them; (b) **pixel-balance vs image-count** — show subsets equalized on pixel-class balance behave differently from subsets equalized on image count; (c) **global-embedding blindness** — show rare-class-bearing images are not separable in global embedding space (they cluster with their common-class scene), so no global selector can prefer them. **This section is the difference between a paper and a null result.**
3. **The representation study (Contribution 3).** Which representation selects best — even if none beats random, "dense self-supervised features rank images differently from classification features, and here is how" is a standalone, citable finding.
4. **The honest method ablation.** Your rarity-weighted patch-coverage method included as a row that *also* doesn't beat random — reframed as *supporting* the mechanism: "even an explicitly rare-class-aware objective cannot overcome the co-occurrence structure, confirming the diagnosis." The failed method becomes evidence *for* Contribution 2, not a hole.

**What makes it publishable vs not (repeat of the one real risk):** a null result that merely restates "random is hard to beat" (already known from CCS in classification) is **thin**. The paper is carried entirely by the *segmentation-specific mechanism* (item 2). No mechanism → not publishable. Mechanism → a genuine "the classification playbook does not transfer to dense prediction, and here's why" contribution that saves the field wasted effort.

**Distinctness from the land-cover coreset paper:** they report *that* selection works for (remote-sensing) segmentation; you explain *when and why it does or doesn't* for natural images, with a rare-class mechanism they never examined. Different genre (characterization vs benchmark), different domain, non-overlapping claim.

**Venue read:** this is a natural **workshop** paper (data-centric AI / limited-data / efficient-CV). A strong mechanism + two-dataset rigor could reach a main-conference short/finding track, but target the workshop.

---

## 14. Risk gates and pivots (decide fast, don't sink compute)

| Risk | Detector | Pivot |
|---|---|---|
| Random unbeatable overall | Stage 1 | **Not a pivot — this is an anticipated outcome.** Ship the §14b no-method paper (Contributions 1–3 + mechanism). |
| No rare-class collapse at all | Stage 1 | mechanism becomes "why selection is neutral for segmentation"; still a §14b characterization paper via Contribution 3 + a different mechanism |
| Recipe non-discriminative | Day-1 check | raise `epochs`, re-freeze |
| Encoder leakage (select+train same backbone inflates) | Stage 2 cross-representation | report cross-representation numbers as the honest result |
| Segmentation resists pruning below ~20% | Stage 1 curve | headline at 20–30%, discuss the floor honestly |
| Compute overrun | budget tracker | run lean core, 2 seeds, drop 5%/50% first pass |
| **Mechanism absent** (no explanation found for the non-transfer) | end of Stage 2 | **the real danger.** Invest here before writing — a null result without a mechanism is the one non-publishable outcome (§0, §14b). |

---

## 15. Week-by-week

- **Week 1:** build system (Stage-0 plumbing on CamVid) + Stage 1 kill-shot on Cityscapes. **Decision gate hit by end of week.**
- **Week 2:** Stage 2 (representation study) + start Stage 3 (method + ablation).
- **Week 3:** finish Stage 3 + Stage 4 (generalization). Draft Fig 1 and Tables 1–4.
- **Week 4:** ADE20K (if time), backfill seeds/ratios, write-up, polish figures, submit.

---

## 16. First actions (today)

1. Stand up the repo skeleton (§12) and the `runs.csv` logger.
2. `00_extract_features.py` for CamVid with `dinov2_patch` + `dinov2_cls`.
3. Implement `random`, `kmeans_centers`, `patch_coverage` behind the common selector interface.
4. Wire the frozen `cityscapes_proxy` recipe + SegFormer-B0 trainer + eval (mIoU, rare-class mIoU).
5. Run the **discriminativeness check** on CamVid, then Stage 0.
6. Move to Cityscapes and fire **Stage 1**. Let the rare-class result choose the paper.

---

### Appendix A — what would NOT be enough (keep yourself honest)
"DINOv2 → cluster → select for segmentation" = ELFS/ZCore ported; not a paper. "Patch instead of global" alone = one ablation. A workshop paper needs the **diagnostic result** (rare-class collapse) **plus** the simple fix, with proper seeds/variance and matched compute — and the fix must be shown to beat **`zcore`**, the strongest label-free *global*-coverage method, or the "dense beats global" claim is unproven. A main-conference version additionally needs the cross-architecture generalization to be clean and the representation-transfer finding to be surprising.

### Appendix B — baseline decision log (why each in/out)
- **Recency is a within-axis tiebreaker, not an admission rule** (§5.0). We compare *mechanisms re-run on our data*, never other papers' reported classification numbers.
- **In & competing:** `random` (floor), `prototypicality` (representativeness, canonical), `kcenter` (global coverage, canonical), `semdedup` (redundancy, canonical), `bpp` (intrinsic, recent-but-weak), **`zcore`** (global FM-subspace coverage, recent-strong — the direct foil).
- **Optional:** `elfs` — overlaps the representativeness axis and is a heavier pseudo-label pipeline; include only with clean code + spare time.
- **Out of track by construction:** the score/difficulty family (EL2N, GraNd, Forgetting, InfoBatch) needs labels/gradients/a trained model. Stating this exclusion is itself a scoping contribution (§5.2). `class_balanced`/`infobatch`/`entropy_warm` appear only as fenced-off labeled references.
- **Folded:** `kmeans_centers` merged into `prototypicality` (same point twice) and now lives only as the "global" arm of the method ablation.
