# Literature check for Proposals 1, 2, 3 (2026-10-11)

Is anything in `docs/proposals-explained.md` already published, and is what's left novel
enough to try? Based on web search (arXiv, CVF, NeurIPS/ICLR proceedings) on 2026-10-11.

**Limits of this check:** it's a targeted search, not a systematic review. Several papers were
read from their abstract or an extracted summary, not in full; the "Read" column says which.
Verify any detail before citing it in the paper.

## Bottom line

| | Taken? | Closest prior work | What is still yours | Verdict |
|---|---|---|---|---|
| **Proposal 1**: dense prototypicality (patch tail score) | **Partly.** "Use local patch features, not a global vector, to pick images for segmentation" is published (FreeSel, NeurIPS 2023). "Score an image by its most unusual patches" is the standard PatchCore anomaly score | FreeSel; PatchCore | The prototypicality / atypicality score at patch level; the controlled CLS → mean → tail comparison; the per-class trade-off analysis | **Low novelty as a method. Still worth running as the mechanism ablation**, not the headline |
| **Proposal 2**: rarity-weighted patch-concept coverage | **Not as specified**, but the closest work is close. OFDS (2024/25) does class-balanced coverage of objects on Cityscapes, beats FreeSel and prototypicality, and looks at the smallest classes, **but needs the target class names** and an open-vocabulary detector | OFDS; FreeSel; concept-coverage pruning for image-caption data (MCL, 2026) | No class names, no detector; covers "stuff" classes too; rarity weights + saturation from unsupervised patch concepts | **Medium novelty; the best remaining bet.** It has to beat FreeSel, and be argued against OFDS |
| **Proposal 3**: region-level, labeled-pixel budget | **Mostly.** Region/pixel selection is well studied, and "random regions beat random whole images at equal pixel budget" is already shown (Block Annotation 2019, PixelPick 2021). A NeurIPS 2025 method has a label-free first stage that picks pixels from foundation-model features | CEREALS, superpixel AL, RIPU, PixelPick, Block Annotation, Kim et al. NeurIPS 2025 | Only the narrow version: **one-shot, cold-start** region selection with a rarity/coverage objective, no model in the loop | **Low–medium novelty, highest cost. Defer** |

**The most important finding isn't about a single proposal.** FreeSel and OFDS already show
that label-free, one-shot selection beats random for Cityscapes segmentation, and both argue
that **local/object features beat global image features** for dense tasks. That overlaps with
the plan's Contribution 1 (§0) and with the "dense beats global" story. Two consequences:

1. **FreeSel must be added as a baseline.** It is label-free, one-shot, uses dense ViT
   features, has public code (`github.com/yichen928/FreeSel`), and has been run on Cityscapes
   segmentation. A reviewer will ask for it before `zcore`.
2. **The paper's novelty has to move** away from "dense beats global" (now prior art) to
   things neither paper does:
   - the **per-class trade-off** and its mechanism (keep-hard drops street furniture in
     ordinary scenes);
   - a **controlled locality spectrum** (CLS → patch mean → patch tail) under one selector;
   - **class-name-free rarity-aware coverage** (Proposal 2).

---

## Proposal 1: dense prototypicality

### Closest work

**FreeSel: "Towards Free Data Selection with General-Purpose Models"** (Xie et al.,
NeurIPS 2023). [arXiv 2309.17342](https://arxiv.org/abs/2309.17342). *Read: method and
Cityscapes sections in full.*
- Takes a DINO ViT (DeiT-S), keeps each image's patch tokens with the highest CLS attention
  (up to an attention-mass ratio τ), spectrally clusters them **within each image** into K
  "semantic patterns", and averages each cluster.
- Selects images by **k-means++-style D² sampling over the pool of semantic patterns**:
  sample a pattern with probability ∝ squared distance to the nearest already-selected
  pattern, then add its whole image.
- Single pass, no labels, no training. Run on PASCAL VOC (detection), **Cityscapes
  segmentation** (DRN model, 10–40%) and classification; beats random and several AL methods.
- Motivation, in their words: "complex scenes are hard to represent globally", the same
  argument as our mechanism (c).
- **Differences from Proposal 1:** FreeSel optimizes *diversity*, not *atypicality*. It has no
  prototypes, no rarity, and no tail score. Its **attention filter keeps salient regions and
  drops the rest**, which may throw away exactly the small traffic lights and poles we care
  about. That is a testable difference and a possible point in our favor.

**PatchCore** (Roth et al., CVPR 2022, industrial anomaly detection).
- Scores each patch by distance to a memory bank of normal patches. **The image score is the
  maximum patch score.**
- That is Proposal 1's aggregation (top-q tail) in another field, so "score an image by its
  most anomalous patches" is not new. Cite it as the source of the aggregator.
- Top-k pooling of patch scores is also standard in weakly-supervised segmentation and
  multiple-instance learning.

**Supporting evidence for the diagnosis:** Singh, "Leveraging Perceptual Scores for Dataset
Pruning in Computer Vision Tasks" (CVPRW 2024).
[arXiv 2408.07243](https://arxiv.org/abs/2408.07243). *Read: summary.*
- On VOC and ADE20K segmentation: "prototypicality score does poorly for semantic
  segmentation because it exaggerates the class imbalance problem."
- This is the `bpp` baseline already in the plan. It supports Contribution 2's per-class
  story, but note that our CLS keep-hard *does* beat random on Cityscapes overall mIoU. That
  difference is worth a sentence in the paper.

### Verdict

- **Not novel as a method.** "Patch-level prototype distance + top-q aggregation" combines
  Sorscher's prototypicality with PatchCore's scoring. A reviewer would call it a natural
  extension.
- **Still worth running**, for two reasons:
  1. It completes a **controlled locality spectrum under one fixed selector**
     (CLS → mean → tail). FreeSel changes the representation *and* the selector at once, so
     it can't isolate locality.
  2. It's cheap and reuses the existing selector.
- Present it as the **mechanism ablation**, not the contribution. Add **FreeSel as a row
  next to it**: if P1's tail beats FreeSel on traffic light / pole, the attention-filter
  argument above becomes a concrete finding.

---

## Proposal 2: rarity-weighted patch-concept coverage

### Closest work

**OFDS: "Single-Pass Object-Focused Data Selection"** (Popp, Zhang, Metzen, Hein, Schott;
arXiv 2412.10032, v2 June 2025; workshop version at ICLR 2025 as "Foundation Model-Based Data
Selection for Dense Prediction Tasks"). [arXiv](https://arxiv.org/abs/2412.10032). *Read:
method, Cityscapes results and limitations, from an extracted summary.*
- Grounding DINO proposes objects **from the target class names**. SAM 2 gives object
  features. Then **per-class k-means**, picking the object nearest each centroid, so the
  subset "semantically covers all target classes" in a class-balanced way.
- Budget counted in annotation units (boxes/masks), not only images.
- **Cityscapes semantic segmentation** (Segmenter ViT-T, 5–30% budgets):

  | Budget | OFDS | Random | FreeSel |
  |---|---|---|---|
  | 5% | 61.0 | 55.7 | 56.0 |
  | 10% | 67.1 | 62.4 | 64.7 |
  | 30% | 71.6 | 70.3 | 70.9 |

- Baselines include FreeSel, k-centers, ELFS, UP-DP, HaCON and **prototypes**. They report
  that image-level methods "fail to consistently outperform random", and they analyze the
  smallest classes.
- **Limitations (their own):** it needs "a set of target class names", and it inherits the
  object proposer's biases. Amorphous "stuff" (road, vegetation, sky, fence, wall) isn't
  naturally an object proposal; the paper doesn't say how they handle it.

**FreeSel** (above): coverage/diversity over local patterns, but **unweighted**. A rare
pattern and a common one count the same, and nothing saturates.

**Rarity-weighted concept coverage elsewhere:**
- **MCL, "Mapping the Concept Landscape"** (Wu & Ma, arXiv 2608.22858, Aug 2026):
  "greedy concept-coverage maximization ... maximize the marginal gain of high-value,
  under-represented concepts." The same objective shape as Proposal 2, but for **image-caption
  pruning** with concepts from scene graphs. *Read: abstract.* This is new and very close in
  spirit; cite it, and don't claim the objective itself is new.
- **Complement Submodular Information** (Iyer, arXiv 2605.24779, May 2026): submodular
  selection designed to keep rare/tail structure while suppressing outliers. Generic data
  selection. *Read: abstract.*
- **VLMine** (WACVW 2025): rarity from VLM keyword frequency, for long-tail mining in
  driving data.
- **Balanced concept curation via hierarchical k-means** (DINOv2/DINOv3 data curation, Vo et
  al. 2024): balanced coverage of global-image concepts at web scale.

### Verdict

- **Not taken as specified.** No paper found uses **unsupervised patch-level concepts** with
  **rarity weights and saturating coverage** for **one-shot segmentation selection without
  class names**. The objective's ingredients (weighted coverage, IDF-style rarity, greedy
  submodular selection) are all known. The novelty is the **dense, class-name-free
  instantiation for segmentation** plus the per-class mechanism.
- **The bar has risen.** OFDS gets large Cityscapes gains *with* class names. Positioning:
  - P2 is the **class-name-free** counterpart. It needs no vocabulary, no detector, and covers
    stuff classes. That matters for unknown or open-ended label spaces, and for "stuff"
    classes such as fence and wall, which an object detector doesn't propose.
  - Report OFDS as a **reference row that uses class names** (out of track, like plan §5.2),
    if its code or a faithful reimplementation is feasible. Otherwise cite its numbers with
    the caveat that setups differ (Segmenter ViT-T vs SegFormer-B0).
- **Minimum to be publishable:** P2 beats **FreeSel** and **`zcore`** (both label-free and
  class-name-free) on rare-class / street-furniture IoU, and the rarity ablation (α = 0 vs
  α > 0) shows that rarity weighting is what helps. Without the α ablation, P2 looks like
  "FreeSel with a different coverage function."
- **Still the best of the three to pursue.**

---

## Proposal 3: region-level selection at a matched pixel budget

### Closest work

**Region-based active learning (multi-round, model in the loop):**
- **CEREALS** (Mackowiak et al., BMVC 2018): region selection with a cost model on
  Cityscapes; 17% of the annotation effort for 95% of full performance.
- **Revisiting Superpixels for AL** (Cai et al., CVPR 2021); **Adaptive Superpixel for AL**
  (Kim et al., ICCV 2023); **RIPU** (Xie et al., CVPR 2022, region impurity + uncertainty);
  **ESA** (2024, entity/superpixel annotation).
- All of them need a trained model to score regions, so none is cold-start.

**Pixel- and block-level budgets:**
- **PixelPick** (Shin et al., ICCVW 2021): studies **annotation diversity at a fixed pixel
  budget**, i.e. a few pixels across many images vs dense labels on a few images. That is the
  "random regions vs random images" question.
- **Block Annotation** (Lin, Upchurch, Bala, ICCV 2019): labeling sub-image blocks instead of
  full images. **12% of pixels annotated as blocks reaches 98% of dense-annotation
  performance; 50% matches it.** *Read: search abstract only; block-selection details not
  verified.*
- So **"regions beat whole images at equal pixel budget" is already known**. In Proposal 3
  it's a sanity check, not a finding.

**Label-free first stage:** Kim, Bae, ..., Sutherland, Noh, "Diffusion-Driven Two-Stage Active
Learning for Low-Budget Semantic Segmentation" (**NeurIPS 2025**,
[arXiv 2510.22229](https://arxiv.org/abs/2510.22229)). *Read: method summary.*
- Stage 1 picks representative **pixels** from diffusion-model features (MaxHerding), with
  no labels. Stage 2 uses a trained head plus diffusion noise disagreement.
- 10 AL rounds; CamVid, ADE-Bed, Cityscapes, Pascal-Context; extremely small pixel budgets.
- **No rare-class analysis** and **no image-vs-pixel comparison at matched budget.**

### Verdict

- **Mostly taken.** Region units, pixel budgets and the region-vs-image gain are all
  published. Representation-based (label-free) pixel picking exists as stage 1 of an AL
  method.
- **What remains:** a **one-shot, cold-start** region selector with an explicit
  **rarity/coverage objective** (Proposal 2 applied to regions), evaluated on rare classes.
  That is a narrow increment over the NeurIPS 2025 paper and the region-AL literature, at the
  highest engineering cost (masked training, step matching, crop sampling).
- **Recommendation: defer further than before.** Consider it only as a short extension
  ("P2's objective applied to regions") if P2 works, and cite Block Annotation / PixelPick for
  the region-vs-image effect instead of re-establishing it.

---

## What this changes in the plan

1. **Add FreeSel as a baseline now** (public code, label-free, one-shot, already run on
   Cityscapes segmentation). Run it on the Tier-1 schedule next to `zcore`: 0.1/0.2/0.3 ×
   3 seeds, about 3 GPU-h.
2. **Add OFDS** as a class-name-informed reference row (out of track), or at least cite its
   Cityscapes numbers.
3. **Reword the novelty claims** in plan §0 and Contribution 1. "Image-level methods fail to
   beat random for segmentation" and "local beats global" are already argued by FreeSel and
   OFDS. Our distinct claims are:
   - the per-class trade-off and its mechanism;
   - the controlled locality spectrum under one selector;
   - class-name-free rarity-aware dense coverage;
   - SegFormer + two datasets + seed bands (FreeSel and OFDS report means over 3 selections,
     without the variance-band rigor of plan §7).
4. **Revised order:** FreeSel + `zcore` baselines → Proposal 1 as the ablation → Proposal 2
   as the method → Proposal 3 only as an extension.

## Sources

- [FreeSel, NeurIPS 2023 (arXiv 2309.17342)](https://arxiv.org/abs/2309.17342) · [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/047682108c3b053c61ad2da5a6057b4e-Abstract.html)
- [OFDS, Single-Pass Object-Focused Data Selection (arXiv 2412.10032)](https://arxiv.org/abs/2412.10032) · [ICLR 2025 workshop version](https://mlanthology.org/iclrw/2025/popp2025iclrw-foundation/)
- [Diffusion-Driven Two-Stage Active Learning, NeurIPS 2025 (arXiv 2510.22229)](https://arxiv.org/abs/2510.22229)
- [Leveraging Perceptual Scores for Dataset Pruning, CVPRW 2024 (arXiv 2408.07243)](https://arxiv.org/html/2408.07243v1)
- [PatchCore overview](https://www.emergentmind.com/topics/patchcore)
- [Mapping the Concept Landscape (arXiv 2608.22858)](https://arxiv.org/abs/2608.22858)
- [Complement Submodular Information Measures (arXiv 2605.24779)](https://arxiv.org/abs/2605.24779)
- [VLMine, WACVW 2025](https://openaccess.thecvf.com/content/WACV2025W/LLVMAD/html/Ye_VLMine_Long-Tail_Data_Mining_with_Vision_Language_Models_WACVW_2025_paper.html)
- [DINOv3 (hierarchical k-means data curation)](https://arxiv.org/abs/2508.10104)
- [CEREALS, BMVC 2018](https://hci.iwr.uni-heidelberg.de/node/6441)
- [Revisiting Superpixels for Active Learning, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Revisiting_Superpixels_for_Active_Learning_in_Semantic_Segmentation_With_Realistic_CVPR_2021_paper.pdf)
- [Adaptive Superpixel for Active Learning, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Adaptive_Superpixel_for_Active_Learning_in_Semantic_Segmentation_ICCV_2023_paper.pdf)
- [ESA: Annotation-Efficient Active Learning for Semantic Segmentation (arXiv 2408.13491)](https://arxiv.org/pdf/2408.13491)
- [PixelPick, ICCVW 2021 (arXiv 2104.06394)](https://arxiv.org/abs/2104.06394)
- [Block Annotation, ICCV 2019](https://mlanthology.org/iccv/2019/lin2019iccv-block/)
- [ZCore (arXiv 2411.15349)](https://arxiv.org/abs/2411.15349)
- [Training-Free Dataset Pruning for Instance Segmentation, ICLR 2025 (uses labels; not label-free)](https://iclr.cc/virtual/2025/poster/28161)
- [Dataset-Aware Cold-Start Active Learning (CSCS), 3D medical segmentation (arXiv 2606.20765)](https://arxiv.org/html/2606.20765v1)
