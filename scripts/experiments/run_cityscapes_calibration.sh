#!/usr/bin/env bash
# Cityscapes recipe calibration (§4, §9): LR sweep + discriminativeness check.
# Run ONCE, before freezing cityscapes_proxy.yaml for the real Stage-1 matrix — these
# runs exist only to validate the frozen recipe, they are not part of the matched-method
# comparison itself and should not be reported as selection-method results.
#
# STATUS (2026-09-27): Parts A and B below are COMPLETE and commented out (results are
# in results/runs.csv / W&B, referenced in cityscapes_proxy.yaml's comments). Outcome:
#   - LR sweep winner: lr=1e-4 (mIoU 72.78% vs 71.43% @ 6e-5 vs 68.78% @ 3e-5) — recipe's
#     `lr` has been updated to 1.0e-4 to match.
#   - Discriminativeness check at the OLD lr=6e-5: full (mIoU 71.43%) vs random-10% mean
#     (mIoU ~47.0% ± ~2.0) — full wins by ~24 points, ~12x the random baseline's own std.
#     Clear pass.
# Part C (active) reruns the discriminativeness check at the NEW frozen lr=1e-4, for
# internal consistency (the check should validate the recipe's actual default, not a
# value that's since been superseded). Cheap — reuses the existing subset files (subset
# selection doesn't depend on lr) and each run only takes ~12 GPU-minutes.
#
# Prereqs: clean git tree (commit first, so runs.csv's git_commit column stays
# trustworthy — this script does NOT pass --allow-dirty on purpose); Cityscapes rare-class
# list already computed (results/rare_classes/cityscapes.json).
#
# results/runs.csv changing after each run does NOT count as "dirty" (git_utils.is_dirty
# excludes it — it's pure output, already self-describing via its own git_commit column).
# Any NEW subset file created by 02_select.py DOES still count (real input) and gets
# committed immediately after creation, before the run that uses it.

set -euo pipefail
cd "$(dirname "$0")/.."

# ============================================================
# PART A (COMPLETE, archived) — LR sweep {3e-5, 6e-5, 1e-4}, full data, seed 0.
# Winner: 1e-4. cityscapes_proxy.yaml's `lr` now reflects this. Uncomment to redo/extend
# (e.g. add a 4th point like 2e-4 to check whether the trend keeps improving).
# ============================================================

# echo "=== LR sweep (full Cityscapes, 100 epochs, seed 0) ==="
#
# python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
#     --set recipe.lr=3e-5 --run-name cs_lrsweep_lr3e-5_full_s0
#
# python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
#     --set recipe.lr=6e-5 --run-name cs_lrsweep_lr6e-5_full_s0
#
# python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
#     --set recipe.lr=1e-4 --run-name cs_lrsweep_lr1e-4_full_s0

# ============================================================
# PART B (COMPLETE, archived) — discriminativeness check at the OLD default lr=6e-5.
# Passed clearly (see STATUS above). Kept only for the record; Part C below is the
# check that actually matters now that the recipe's default lr has changed.
# ============================================================

# echo "=== Discriminativeness check @ lr=6e-5 (random 10%, 3 seeds) ==="
#
# for seed in 0 1 2; do
#     subset="results/subsets/cityscapes_random_0.1_${seed}.json"
#     python scripts/02_select.py --dataset cityscapes --selection random --ratio 0.1 --seed "$seed"
#     git add "$subset"
#     git commit -q -m "Add cityscapes random 10% subset, seed ${seed} (calibration script)"
#     python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
#         --set recipe.lr=6e-5 --subset "$subset" --run-name "cs_disccheck_random_r0.1_s${seed}"
# done

# ============================================================
# PART C (ACTIVE) — rerun the discriminativeness check at the NEW frozen lr=1e-4.
# cs_lrsweep_lr1e-4_full_s0 (Part A) is already the full-data baseline at this lr — not
# retrained here. Reuses the subset files Part B already created+committed (selection is
# independent of lr, so the same 3 subset files are still valid — no need to re-select).
# ============================================================

echo "=== Discriminativeness check @ lr=1e-4 (random 10%, 3 seeds) ==="

for seed in 0 1 2; do
    subset="results/subsets/cityscapes_random_0.1_${seed}.json"
    python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
        --subset "$subset" --run-name "cs_disccheck_random_r0.1_s${seed}_lr1e-4"
done

echo "=== Done ==="
echo "Compare cs_lrsweep_lr1e-4_full_s0 against cs_disccheck_random_r0.1_s{0,1,2}_lr1e-4"
echo "in results/runs.csv (or W&B) — full should still clearly beat random's mean+std."
echo "results/runs.csv itself is still uncommitted (by design, see header) — commit it once"
echo "you're done reviewing all rows: git add results/runs.csv && git commit -m '...'"
