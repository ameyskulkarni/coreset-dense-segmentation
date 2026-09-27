# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Training/eval system for a research project on label-free coreset selection for semantic
segmentation (SegFormer-B0 on Cityscapes/ADE20K/CamVid). The research plan, narrative, and
full experiment ladder live in `segmentation-coreset-experiment-plan.md` — read it before
making any methodology decision (recipe values, ratios, seed counts, baseline choices);
this repo implements its §12 spec. `docs/adr/0001-epoch-based-training.md` is the
authoritative record of *why* training is epoch-based (not iteration-based) and how the LR
schedule/epoch counts were chosen — read it before changing anything in `configs/recipe/`.

There is no test suite, linter, or CI configured in this repo. Don't invent commands for
these — if asked to add them, ask what tooling is wanted first.

## Environment & commands

Conda env `coreset-dense-segmentation` (already exists on this machine; `conda activate
coreset-dense-segmentation`), then `pip install -e .`. First run of any script downloads
pretrained weights (SegFormer-B0 encoder, DINOv2, ResNet-50, ...) from HF Hub/torch.hub —
needs internet once. Expect an "unauthenticated requests to the HF Hub" warning on every
run; it's a rate-limit notice, not an error.

Dataset roots default to `/home/cognition/datasets/{cityscapes,ADEChallengeData2016,camvid}`,
overridable via `CITYSCAPES_ROOT`/`ADE20K_ROOT`/`CAMVID_ROOT` env vars (see
`configs/dataset/*.yaml`).

Pipeline (one script per stage, `scripts/00-04_*.py`):
```bash
python scripts/00_extract_features.py --dataset cityscapes --features dinov2_vits14   # only needed for feature-based selectors
python scripts/01_compute_rare_classes.py --dataset cityscapes                        # once per dataset, freezes bottom-K rare classes
python scripts/02_select.py --dataset cityscapes --selection random --ratio 0.1 --seed 0
python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
    --subset results/subsets/cityscapes_random_0.1_0.json
python scripts/04_eval.py --run-dir results/checkpoints/<run_name>                     # re-score a checkpoint without retraining
```
Every script takes three composable config sources (later wins): `--dataset/--model/--recipe/--selection/--features <name>` (loads `configs/<group>/<name>.yaml`), `--experiment path.yaml` (names fragments + direct field overrides), `--set key.path=value` (highest-precedence dotlist override). See `segcoreset/utils/config.py`.

`03_train.py` **refuses to run on a dirty git tree** (provenance: `git_commit` in the results ledger must be trustworthy) unless `--allow-dirty` is passed. `results/metrics/` and `scripts/experiments/` are gitignored specifically so iterating on ledgers/orchestration scripts doesn't trip this — everything else (code, `configs/`, `results/subsets/*.json`, `results/rare_classes/*.json`) is a real input and should stay committed before a run that depends on it.

Ad hoc multi-run experiments (LR sweeps, discriminativeness checks, baseline sweeps across ratios/seeds) live as bash scripts under `scripts/experiments/*.sh` — sequential `python scripts/0X_*.py` calls with descriptive `--run-name`s, not part of the numbered pipeline. This directory is gitignored (scratch orchestration, not the source of truth for a run's actual hyperparameters — that's captured per-run regardless, see Provenance below). Follow this pattern rather than inventing a new one when asked to script a batch of runs.

## Architecture

**Config composition** (`segcoreset/utils/config.py`): a run's config is assembled from named YAML fragments per group (`dataset`, `model`, `recipe`, `selection`, `features`), merged via OmegaConf, then hashed (`config_hash`) for provenance. `Trainer`/`Evaluator`/scripts only ever see the final merged `DictConfig` — they don't know or care which of the three input methods produced it.

**Epoch-based training, not iteration-based** (ADR 0001) — the single most important cross-cutting design decision. `Trainer` (`segcoreset/train/trainer.py`) is the *only* place `epochs`/`warmup_epochs` get resolved to step counts, and it does so using the **realized subset size** (`len(train_ds) // batch_size`), not a config-time constant. A 10% subset therefore trains for ~10% of the full run's gradient steps, not the same step count repeated more — this is deliberate (matches the closest comparable prior work's efficiency framing) and reverses an earlier fixed-iteration design; don't reintroduce iteration-based scheduling without reading the ADR's full reasoning first. Recipes are **per-dataset, not shared** (`configs/recipe/{cityscapes,ade20k,camvid}_proxy.yaml`, plus `*_ceiling.yaml` for one-off long-budget context runs) because the same epoch count implies wildly different absolute compute across datasets of different sizes (ADE20K's full set is ~7x Cityscapes'). LR schedule is restricted to exactly two shapes, `poly` (default; the literature-standard SegFormer/mmseg schedule, power=1.0 is literally linear decay to zero) and `cosine`/`constant` as documented fallbacks — see `segcoreset/train/lr_schedule.py::WarmupLR`.

Recipe hyperparameters (`lr` in particular) are **empirically calibrated per dataset via sweep scripts, not literature defaults left unquestioned** — check a recipe YAML's own header comments for the current frozen value and the sweep evidence behind it before assuming it's wrong or copying it to another dataset.

**Selection ↔ training decoupling**: selectors (`segcoreset/selection/*.py`, common `select(image_ids, features, budget, seed, cfg) -> list[str]` interface) run once, before any training, writing an id-list to `results/subsets/{dataset}_{method}_{ratio:g}_{seed}.json`. Training only ever consumes that id-list — it never knows which selection method produced it. Adding a new selection method is one new file in `segcoreset/selection/` + a config in `configs/selection/`, no trainer changes.

**Provenance chain**: every training run logs `git_commit` + `config_hash` (identifies exact code+config state) + `wandb_run_id`/`wandb_url` (direct link to the W&B dashboard) into `results/metrics/runs.csv`, in addition to full hyperparameter logging to W&B's own config panel. This is why the dirty-tree gate exists and why `results/metrics/` itself is gitignored (it's the mutable *output* of that provenance system, not an input — each row is independently self-describing regardless of whether the ledger file itself is tracked).

**Eval protocol differs by dataset and by when it runs.** Final/standalone eval uses each dataset's real protocol (`configs/dataset/*.yaml`'s `eval.mode`) — `sliding_window` for Cityscapes (tiles at the training crop size with 2/3 overlap, mmseg convention, `segcoreset/eval/sliding_window.py`) since its native resolution exceeds the crop size, `whole_image` (resize) for ADE20K/CamVid. Periodic mid-training eval (`Trainer.train()`'s `eval_every_epochs` hook) always forces cheap `whole_image` + a fixed 100-image subsample regardless of the dataset's real protocol — its numbers are a monitoring trend line, **not directly comparable** to the final reported metric for Cityscapes. Rare-class mIoU (the thesis metric) only populates once `results/rare_classes/<dataset>.json` exists (`01_compute_rare_classes.py`, run once per dataset from full-label pixel frequency — labels used for analysis only, never selection); `Evaluator` silently omits it if that file is absent.

**Model contract**: every architecture in `segcoreset/models/registry.py` (SegFormer-B0, DeepLabV3+, U-Net) implements `forward(pixel_values) -> logits` at input resolution — architecture-specific upsampling/decode-head differences are absorbed inside each wrapper so `Trainer`/`Evaluator` are architecture-agnostic.
