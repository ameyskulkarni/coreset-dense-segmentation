# segcoreset

Training and evaluation system for **label-free coreset selection for semantic
segmentation** (see `segmentation-coreset-experiment-plan.md` for the full research plan —
this repo implements §12 of that document).

Selection methods implemented so far: the `random`/`full` reference points and
`prototypicality` (docs/prototypicality.md). The other baselines and the
aggregate-tables/plotting scripts are not built yet — see "What's not here yet" below.

## Setup

```bash
pip install -e .
```

Uses the conda env `coreset-dense-segmentation` (torch cu121, RTX 3090). First run of any
script downloads pretrained weights (SegFormer-B0 encoder, DINOv2 ViT-B/14, ResNet-50,
...) from HuggingFace Hub / torch.hub — needs internet once, then everything is cached
locally.

Datasets expected on disk (paths overridable via env var or config, see `configs/dataset/`):

| Dataset | Default path | Status |
|---|---|---|
| ADE20K | `/home/cognition/datasets/ADEChallengeData2016` | **on disk, ready to use** |
| Cityscapes | `$CITYSCAPES_ROOT` or `/home/cognition/datasets/cityscapes` | pending approval |
| CamVid | `$CAMVID_ROOT` or `/home/cognition/datasets/camvid` | not yet downloaded |

## Repo layout

```
configs/            # one YAML per dataset/model/recipe/selection/features + experiment configs
segcoreset/
  data/              # dataset wrappers (Cityscapes/ADE20K/CamVid), joint transforms, rare-class freq
  features/          # DINOv2/DINOv3/ResNet-50/SegFormer-encoder/CLIP extractors + on-disk cache
  models/            # SegFormer-B0, DeepLabV3+, U-Net — common forward(pixel_values)->logits contract
  selection/         # common select() interface; random, full, prototypicality/ implemented, rest drop in as new files
  train/             # fixed-epoch Trainer (loss, LR schedule, AMP, checkpointing, W&B)
  eval/               # Evaluator (mIoU, rare-class mIoU, boundary F, pixel acc), sliding-window inference
  logging/           # results/metrics/runs.csv ledger (append-only) + thin W&B wrapper
  utils/             # config composition, seeding, git provenance
scripts/             # one script per pipeline stage, see below
results/
  metrics/           # runs.csv (append-only ledger, §12.3 schema) + ad hoc exports —
                     # gitignored on purpose: a mutable scratch/output area, edit freely
                     # without needing a commit first; each row self-describes its own
                     # provenance via git_commit/config_hash/wandb_url
  subsets/           # selected image-id lists (json), tracked in git — cheap, reproducibility-critical
  rare_classes/       # frozen bottom-K rare-class lists per dataset, tracked in git
  features/          # cached embeddings (gitignored — regenerate from configs)
  checkpoints/       # per-run dir: resolved config.yaml + last.pt/final.pt (gitignored)
```

## Config system

Every script accepts three ways to configure a run, composed in this order (later wins):

1. `--dataset X --model Y --recipe Z ...` — pick named fragments from `configs/<group>/`.
2. `--experiment path/to/exp.yaml` — a small YAML naming fragments by group
   (`dataset: cityscapes`) plus any direct field overrides (`ratio: 0.2`, `wandb: {...}`).
   This is the primary "one file per experiment" workflow — see `configs/experiment/`.
3. `--set key.path=value ...` — dotlist overrides for one-off tweaks, highest precedence.

The resolved config is hashed (`config_hash`) and saved next to every run's checkpoint for
exact reproducibility.

## Pipeline

```bash
# 1. cache DINOv2 features (needed once you add feature-based selectors; random/full skip this)
python scripts/00_extract_features.py --dataset ade20k --features dinov2_vitb14

# 2. freeze the rare-class list from full-train pixel frequency (once per dataset)
python scripts/01_compute_rare_classes.py --dataset ade20k --k 15

# 3. select a subset (label-free, before any training)
python scripts/02_select.py --dataset ade20k --selection random --ratio 0.2 --seed 0
#    feature-based selectors (e.g. prototypicality) read saved embeddings — full walkthrough
#    in docs/prototypicality.md:
python scripts/00b_derive_embeddings.py --dataset cityscapes --features dinov2_vitb14 --method cls
python scripts/02_select.py --dataset cityscapes --selection prototypicality-hard --ratio 0.2 --seed 0

# 4. train (fixed-epoch recipe; periodic + final eval; logs to W&B; appends to runs.csv)
python scripts/03_train.py --dataset ade20k --model segformer_b0 --recipe ade20k_proxy \
    --subset results/subsets/ade20k_random_0.2_0.json

# 4b. or drive the same run from one experiment file:
python scripts/03_train.py --experiment configs/experiment/example_ade20k_random20.yaml \
    --subset results/subsets/ade20k_random_0.2_0.json

# 5. evaluate a checkpoint standalone (independent of training), appends a new runs.csv row
python scripts/04_eval.py --run-dir results/checkpoints/<run_name>
```

`03_train.py` refuses to run on a dirty git working tree (so `runs.csv`'s `git_commit`
column is trustworthy) — commit your changes first, or pass `--allow-dirty` for a quick
smoke test.

## Metrics (§7)

Computed in one eval pass, both mid-training (fast, whole-image, subsampled) and in the
final/standalone pass (full protocol — sliding-window for Cityscapes, per `configs/dataset/*.yaml`):
mIoU, rare-class mIoU (against the frozen bottom-K list), per-class IoU, boundary F-score
(trimap, 3px tolerance), pixel accuracy, wall-clock GPU-hours.

## Frozen training recipe (§4, §9, ADR 0001 — docs/adr/0001-epoch-based-training.md)

The recipe (LR, schedule, batch size, epoch cap) is tuned once on the full dataset and
frozen — identical across every selection method, ratio, and seed, so subset *content* is
the only variable (and, deliberately, the resulting wall-clock time — see the ADR). See
`configs/recipe/cityscapes_proxy.yaml` / `ade20k_proxy.yaml` / `camvid_proxy.yaml` for the
exact per-dataset values (AdamW, lr=1e-4 for Cityscapes (calibrated by LR sweep), poly decay with a short linear warmup, weight
decay 0.01, AMP fp16). `Trainer` is the only place `epochs`/`warmup_epochs` get resolved to
step counts, using the realized subset size — nothing else in the codebase reasons about
iterations. Before running the full ladder, run the **discriminativeness check** (§4):
confirm full-100%@N-epochs clearly beats 10%-random@N-epochs; if they tie, raise `epochs`
and re-freeze.

## What's not here yet

- Label-free selection methods beyond `random`/`full`/`prototypicality` (k-center,
  SemDeDup, bpp, ZCore, `patch_coverage`) — drop in as new modules under
  `segcoreset/selection/` (see `selection/base.py` for the interface) once features are cached.
- `05_aggregate.py` / `06_plots.py` (paper tables/figures, retention computation) and a
  `make killshot` / `make lean_core` batch-launcher — planned once the baseline methods land.
