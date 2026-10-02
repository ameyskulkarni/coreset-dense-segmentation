# Experiment Results Log

Running record of finished experiments. Append a new `## Experiment N` section per batch; keep the conventions below so tables stay comparable.

**Conventions**
- Source of truth: `results/metrics/runs.csv` (cross-checked against W&B exports in `results/metrics/wandb_*_export_*.csv`).
- All metrics are **final eval** numbers (Cityscapes: sliding-window, 500 val images), in **%**. Mid-training `val/miou` in W&B is a cheap whole-image 100-image trend line and is *not* comparable to these.
- Rare-class mIoU = mean IoU over the frozen bottom-5 classes (`results/rare_classes/cityscapes.json`).
- `mean ± std` is across seeds, **sample std (ddof=1)**, n = 3 unless stated.
- Retention = mean mIoU(subset) / mean mIoU(full-100%, same recipe, same epochs).
- Train GPU-h is `gpu_hours_train` from `runs.csv` (training only; W&B "Runtime" also includes ~0.25–0.45 h of eval).

---

## Experiment 1 — Random-selection baseline, Cityscapes / SegFormer-B0

**Goal.** Establish the random-selection variance band across the ratio ladder before any real selection method is run (plan §5.1, §7–8). Every later method must clear this band (mean outside random's ±1 std) to count as a win. ratio = 1.0 is the full-data reference (denominator for retention).

**Setup** (script: `scripts/experiments/run_cityscapes_random_baseline.sh`, frozen recipe `configs/recipe/cityscapes_proxy.yaml`, ADR 0001)
- Model: SegFormer-B0 (`nvidia/mit-b0` init), Cityscapes train 2975 / val 500, 19 classes, crop 512×1024.
- Recipe: epoch-based, 100 epochs, batch 8, AdamW, lr 1e-4, wd 0.01, poly decay (power 1) to 0, 6 warmup epochs, AMP fp16. Steps scale with subset size (ADR 0001), so a 20% subset gets ~20% of the gradient steps.
- Selection: uniform random without replacement. **Seed N fixes both the subset draw and the training RNG** (at ratio 1.0 the seed only varies training stochasticity).
- Seeds: 0, 1, 2.
- Eval: sliding window (stride 341×683), single scale.
- Status: **ratios 1.0, 0.9, 0.7, 0.5, 0.3, 0.2 done (18 runs). Ratios 0.1 and 0.05 not yet run — the launcher script aborted at ratio 0.1 (see note 6).**

### Summary (mean ± std over 3 seeds)

| Ratio | n_img | Steps | mIoU (%) | Rare-class mIoU (%) | Pixel acc (%) | Boundary F (%) | Train GPU-h | Retention (mIoU) |
|---|---|---|---|---|---|---|---|---|
| 1.0 (full) | 2975 | 37100 | 72.61 ± 0.09 | 62.38 ± 0.14 | 95.31 ± 0.01 | 45.07 ± 0.22 | 1.27 ± 0.04 | 100.0% |
| 0.9 | 2678 | 33400 | 72.05 ± 0.13 | 60.63 ± 0.41 | 95.28 ± 0.02 | 44.56 ± 0.34 | 1.14 ± 0.01 | 99.2% |
| 0.7 | 2082 | 26000 | 70.58 ± 0.30 | 58.75 ± 0.53 | 95.05 ± 0.05 | 43.23 ± 0.30 | 0.91 ± 0.00 | 97.2% |
| 0.5 | 1488 | 18600 | 68.60 ± 0.98 | 54.18 ± 2.13 | 94.78 ± 0.07 | 41.52 ± 0.14 | 0.68 ± 0.00 | 94.5% |
| 0.3 | 892 | 11100 | 64.19 ± 0.52 | 44.93 ± 1.74 | 94.17 ± 0.08 | 38.45 ± 0.24 | 0.44 ± 0.00 | 88.4% |
| 0.2 | 595 | 7400 | 60.96 ± 0.48 | 38.65 ± 1.42 | 93.72 ± 0.04 | 35.78 ± 0.51 | 0.32 ± 0.00 | 84.0% |
| 0.1 | – | – | pending | pending | pending | pending | pending | pending |
| 0.05 | – | – | pending | pending | pending | pending | pending | pending |

Observations
- Rare-class mIoU degrades much faster than overall mIoU: at 20% data, mIoU retains 84% but rare-class mIoU only 62% of full (38.65 vs 62.38).
- Seed variance grows as the subset shrinks, and is largest for rare-class mIoU (std 2.13 at 0.5, 1.74 at 0.3, 1.42 at 0.2 vs 0.14 at full). Part of this is subset-draw variance, not just training noise.
- Pixel accuracy barely moves (95.3 → 93.7), so it is not informative for this comparison.

### Per-run results

| Run | Ratio | Seed | n_img | mIoU | Rare mIoU | Pixel acc | Boundary F | GPU-h | W&B |
|---|---|---|---|---|---|---|---|---|---|
| cs_random_r1.0_s0 | 1.0 | 0 | 2975 | 72.52 | 62.23 | 95.31 | 45.32 | 1.31 | [fznjgm5a](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/fznjgm5a) |
| cs_random_r1.0_s1 | 1.0 | 1 | 2975 | 72.60 | 62.52 | 95.33 | 44.97 | 1.25 | [nakm2xix](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/nakm2xix) |
| cs_random_r1.0_s2 | 1.0 | 2 | 2975 | 72.70 | 62.40 | 95.31 | 44.93 | 1.26 | [3qr495hy](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/3qr495hy) |
| cs_random_r0.9_s0 | 0.9 | 0 | 2678 | 72.04 | 60.88 | 95.29 | 44.85 | 1.13 | [8od2mdzg](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/8od2mdzg) |
| cs_random_r0.9_s1 | 0.9 | 1 | 2678 | 71.92 | 60.15 | 95.26 | 44.64 | 1.14 | [rrw3gzqu](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/rrw3gzqu) |
| cs_random_r0.9_s2 | 0.9 | 2 | 2678 | 72.18 | 60.85 | 95.29 | 44.18 | 1.14 | [0ba4ifzy](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/0ba4ifzy) |
| cs_random_r0.7_s0 | 0.7 | 0 | 2082 | 70.55 | 59.17 | 95.06 | 43.55 | 0.91 | [ezvylj2v](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/ezvylj2v) |
| cs_random_r0.7_s1 | 0.7 | 1 | 2082 | 70.31 | 58.16 | 95.00 | 43.18 | 0.91 | [24zrx3yv](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/24zrx3yv) |
| cs_random_r0.7_s2 | 0.7 | 2 | 2082 | 70.90 | 58.93 | 95.10 | 42.95 | 0.90 | [cw452x2g](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/cw452x2g) |
| cs_random_r0.5_s0 | 0.5 | 0 | 1488 | 67.62 | 52.34 | 94.71 | 41.52 | 0.68 | [knw5g79w](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/knw5g79w) |
| cs_random_r0.5_s1 | 0.5 | 1 | 1488 | 68.63 | 53.69 | 94.79 | 41.65 | 0.68 | [6s512m82](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/6s512m82) |
| cs_random_r0.5_s2 | 0.5 | 2 | 1488 | 69.57 | 56.51 | 94.86 | 41.38 | 0.68 | [uhxs7dz4](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/uhxs7dz4) |
| cs_random_r0.3_s0 | 0.3 | 0 | 892 | 64.16 | 45.07 | 94.15 | 38.69 | 0.44 | [q7vs5any](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/q7vs5any) |
| cs_random_r0.3_s1 | 0.3 | 1 | 892 | 63.68 | 43.12 | 94.10 | 38.43 | 0.44 | [46dr6xmw](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/46dr6xmw) |
| cs_random_r0.3_s2 | 0.3 | 2 | 892 | 64.72 | 46.61 | 94.26 | 38.22 | 0.44 | [b7kwlxd9](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/b7kwlxd9) |
| cs_random_r0.2_s0 | 0.2 | 0 | 595 | 60.82 | 37.47 | 93.76 | 36.35 | 0.33 | [ysk3k96m](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/ysk3k96m) |
| cs_random_r0.2_s1 | 0.2 | 1 | 595 | 61.49 | 40.22 | 93.73 | 35.63 | 0.32 | [k4oe2tzd](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/k4oe2tzd) |
| cs_random_r0.2_s2 | 0.2 | 2 | 595 | 60.56 | 38.26 | 93.68 | 35.36 | 0.33 | [pcbmjq8x](https://wandb.ai/amey1695-personal/coreset-dense-segmentation/runs/pcbmjq8x) |

### Calibration context (not part of the baseline table)

Earlier runs that fixed the recipe; kept for reference, see `configs/recipe/cityscapes_proxy.yaml` header.

| Check | Setting | mIoU (%) | Rare mIoU (%) |
|---|---|---|---|
| LR sweep, full data, seed 0 | lr 3e-5 | 68.78 | 54.86 |
| | lr 6e-5 | 71.43 | 59.93 |
| | lr 1e-4 (frozen) | 72.78 | 62.68 |
| Discriminativeness, 10% random, 3 runs | lr 6e-5 | 46.30 / 49.14 / 45.54 | 4.21 / 12.49 / 4.66 |
| | lr 1e-4 | 52.66 / 54.81 / 51.19 | 19.96 / 24.27 / 17.06 |

(The 3 values per row are the runs named `_s0/_s1/_s2`; see the data-integrity note below on their seeds. lr 3e-5 appears twice in `runs.csv` as an identical-config rerun: 68.76 vs 68.78.)

### Data-integrity notes (from the runs.csv vs W&B cross-check, 2026-10-01)

All 28 runs in `runs.csv` match the W&B export one-to-one on run name, git commit, config hash, ratio, epochs, lr, seed, mIoU, rare-class mIoU, pixel acc and boundary F (max abs difference ~1e-16). Nothing was corrected. Things worth knowing:

1. `cs_disccheck_random_r0.1_s1/_s2` (both lr variants) have `seed = 0` in `runs.csv` and `recipe.seed_train = 0` in W&B, so the `_s1/_s2` suffix does not correspond to training seed 1/2. Their subsets (committed as seeds 1/2 in git) may differ, but the training RNG was seed 0. These are calibration runs, not in the table above.
2. `config_hash` does not distinguish ratio/subset: e.g. `995fe9bbcaf3` is shared by r1.0_s0, r0.9_s0, r0.7_s0, … and by the lr 1e-4 full sweep run. The hash varies only with seed (s0/s1/s2), so it cannot tell two runs with different subsets apart. `git_commit` is what separates them.
3. `gpu_hours_train` (runs.csv) is consistently ~0.25–0.45 h below W&B `Runtime`; the gap is the final sliding-window eval (and startup), which `gpu_hours_train` excludes. Expected, not an inconsistency.
4. ~~Stale README lr~~ — README said lr 6e-5 / cosine; fixed to the frozen lr 1e-4 / poly (2026-10-02).
5. `cs_lrsweep_lr3e-5_full_s0` appears twice in both files (reruns at different git commits); metrics differ by ~0.02 mIoU points.
6. **Why 0.1 and 0.05 didn't run:** `run_cityscapes_random_baseline.sh` uses `set -e` and, per ratio/seed, runs `02_select.py` then `git add` + `git commit -q`. The 0.1 subset files (`cityscapes_random_0.1_{0,1,2}.json`) were already committed by the earlier calibration script, and selection is deterministic, so re-selecting rewrote identical files; `git commit` then found nothing to commit, exited 1, and `set -e` killed the script right after the 0.2 seed-2 run (last ledger row 2026-09-28T23:36Z). Nothing was running when checked. Fix: guard the commit (e.g. `git diff --cached --quiet || git commit ...`) and run ratios 0.1 and 0.05 only.

---

<!-- Append new experiments below as "## Experiment N — <title>" -->
