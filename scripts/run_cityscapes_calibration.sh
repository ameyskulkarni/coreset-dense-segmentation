#!/usr/bin/env bash
# Cityscapes recipe calibration (§4, §9): LR sweep + discriminativeness check.
# Run ONCE, before freezing cityscapes_proxy.yaml for the real Stage-1 matrix — these
# runs exist only to validate the frozen recipe, they are not part of the matched-method
# comparison itself and should not be reported as selection-method results.
#
# What this does:
#   1. LR sweep {3e-5, 6e-5, 1e-4} on full data, same frozen epochs=100/poly schedule,
#      varying only lr (§9 rule 1: "tune the recipe once, on full data only").
#      lr=6e-5 is the recipe's current default (literature-standard for SegFormer) and
#      also doubles as the full-data baseline for step 2 below — not trained twice.
#   2. Discriminativeness check (§4, mandatory gate): random 10% subset, 3 seeds, same
#      recipe. Full-data must clearly beat random's mean (+/- std across the 3 seeds) —
#      if it doesn't, the epoch budget can't discriminate selection methods yet and
#      needs raising before any real experiment is trustworthy.
#
# After this finishes: compare mIoU across the 3 cs_lrsweep_* rows in results/runs.csv
# to confirm/replace the default lr, and compare cs_lrsweep_lr6e-5_full_s0 against the
# 3 cs_disccheck_random_r0.1_s* rows for the discriminativeness gate. If a different lr
# clearly wins the sweep, redo the discriminativeness check with that lr before treating
# the recipe as frozen.
#
# Prereqs: clean git tree (commit first, so runs.csv's git_commit column stays
# trustworthy — this script does NOT pass --allow-dirty on purpose); Cityscapes rare-class
# list already computed (results/rare_classes/cityscapes.json).
#
# ~7 runs, ~1.3-1.5 GPU-h each -> roughly 9-11 GPU-hours end to end on one RTX 3090.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1/2: LR sweep (full Cityscapes, 100 epochs, seed 0) ==="

python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
    --set recipe.lr=3e-5 --run-name cs_lrsweep_lr3e-5_full_s0

python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
    --run-name cs_lrsweep_lr6e-5_full_s0   # recipe default; also the discriminativeness-check full baseline

python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
    --set recipe.lr=1e-4 --run-name cs_lrsweep_lr1e-4_full_s0

echo "=== 2/2: Discriminativeness check (random 10%, 3 seeds) ==="

for seed in 0 1 2; do
    python scripts/02_select.py --dataset cityscapes --selection random --ratio 0.1 --seed "$seed"
    python scripts/03_train.py --dataset cityscapes --model segformer_b0 --recipe cityscapes_proxy \
        --subset "results/subsets/cityscapes_random_0.1_${seed}.json" \
        --run-name "cs_disccheck_random_r0.1_s${seed}"
done

echo "=== Done ==="
echo "Check results/runs.csv (or W&B) for: cs_lrsweep_lr3e-5_full_s0, cs_lrsweep_lr6e-5_full_s0,"
echo "cs_lrsweep_lr1e-4_full_s0, cs_disccheck_random_r0.1_s{0,1,2}."
