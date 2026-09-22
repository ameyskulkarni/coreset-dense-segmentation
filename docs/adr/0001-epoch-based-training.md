# ADR 0001: Fixed-epoch training, not fixed-iteration; cosine LR schedule

**Status:** Accepted (2026-09-21)
**Supersedes:** the "Fix iterations, NOT epochs" rule in
`segmentation-coreset-experiment-plan.md` §4 (original version), and the poly
LR schedule in `configs/recipe/*.yaml`.

## Context

The original plan (§4) froze every training run at a fixed **iteration**
count (e.g. 20,000 for Cityscapes/ADE20K) regardless of subset size, so a 10%
subset would repeat its images ~10x more often than the full dataset to reach
the same total gradient-step count. The stated reason: this isolates "is the
selected *content* good" from "did this run just get less total compute,"
which matters for a causal/diagnostic claim ("method X's selection quality
caused this mIoU gap").

Two things surfaced that changed the calculus:

1. **A real-world framing objection:** in practice, selecting less data is
   done specifically *to reduce training cost*. A paper that holds compute
   fixed across subset sizes doesn't measure that benefit at all — it
   measures something else (pure content-quality attribution), which is a
   valid but different question than the one motivating coreset selection in
   the first place.
2. **Direct precedent in the closest comparable prior work.** Nogueira et al.
   (arXiv 2505.01225), the paper this project's plan explicitly distinguishes
   itself from (label-free coreset selection for segmentation, U-Net/SegFormer,
   subset curves), states: *"Since all models are trained for a fixed number
   of 100 epochs, the reported per-epoch times allow us to directly quantify
   the computational savings enabled by coreset selection."* That is the
   efficiency framing, treated as a **result to report**, not a confound to
   eliminate — using the fixed-epoch convention specifically to make it
   measurable.

Given (1) and (2), the fixed-iteration convention was reversed.

**One point of confusion resolved during discussion, recorded here because
it's a common misreading of the original rule:** fixing iterations does NOT
mean a pruned subset gets *more* weight updates than full data — the update
count (20,000) was already identical across all subset sizes under the old
rule. What differs between the two conventions is *epoch count* (how many
times each surviving image is revisited), not raw update count. Fixed-epoch
training doesn't change that either — it makes both the epoch count *and* the
per-run compute vary with subset size, together, which is the actual efficiency
claim.

## Decision

**Train every run for a fixed number of epochs, calibrated per dataset**, not
iterations. A 10% subset now does ~10% of the full run's gradient steps and
finishes in ~10% of the wall-clock time — this is the reported efficiency
result, matching the framing in 2505.01225.

Epoch counts are **not** a blind copy of 2505.01225's "100" — their dataset
sizes/hardware/crop resolution are unstated relative to ours, and applying
100 epochs uniformly here would be wildly uneven: ADE20K's full train set
(20,210 images) is ~7x Cityscapes' (2,975) and ~55x CamVid's (367), so the
same epoch count implies wildly different absolute compute per dataset. Each
dataset's epoch count was instead calibrated to land close to the *previous*
iteration budget's total step count on **full data** — those budgets were
already reasoned about relative to this project's single-3090, no-run-over-
~2.5h hardware constraint (plan §Hardware) — so the full-data reference run's
wall-clock is roughly unchanged, and only subset runs get faster (the desired
effect).

| Recipe file | epochs | warmup epochs | full-data steps/epoch | ≈ total steps (full data) | old iteration budget |
|---|---|---|---|---|---|
| `cityscapes_proxy.yaml` | 50 | 3 | 371 | 18,550 | 20,000 |
| `cityscapes_ceiling.yaml` | 160 | 10 | 371 | 59,360 | 60,000 |
| `ade20k_proxy.yaml` | 8 | 1 | 2,526 | 20,208 | 20,000 |
| `ade20k_ceiling.yaml` | 16 | 2 | 2,526 | 40,416 | 40,000 |
| `camvid_proxy.yaml` | 300 | 15 | 22 | 6,600 | 10,000 (deliberately shorter — see below) |

CamVid is the one deliberate exception: it's the Stage-0 pilot whose *numbers*
the plan explicitly discards ("shake out the loop... discard numbers
scientifically, keep the code"), so 300 epochs was chosen for pilot speed
rather than for matching the old convergence depth exactly.

**A fixed-iteration comparison is kept as a secondary appendix ablation**
("does the method ranking change under matched compute instead of matched
epochs?"), not the primary convention — this recovers the original isolation
argument as a robustness check without making it the paper's main framing.

### LR schedule: `cosine`, not `poly`

The poly-decay schedule (power 1.0, i.e. linear decay to 0) is replaced with
exactly two options, per explicit request to avoid "fancy" schedulers:
`constant` (flat `lr` after warmup) and `cosine` (cosine anneal to `min_lr`
after warmup). Implemented in `segcoreset/train/lr_schedule.py::WarmupLR`.

**Recommendation: use `cosine`, with `min_lr: 0.0`, as the default for every
recipe.** Reasoning:
- Cosine anneals to a low LR by the end of training regardless of the exact
  epoch count chosen, which matters here because the epoch counts above are
  *provisional*, pending the discriminativeness check (§4) — cosine is
  forgiving of a schedule that turns out to run a bit long or short, whereas
  constant LR trained too long can plateau/oscillate near a fixed non-zero LR
  and trained too short leaves the model under-annealed either way.
- It is the standard choice for fine-tuning a pretrained transformer encoder
  + randomly-initialized decode head over a bounded number of epochs — the
  regime here is closer to "fine-tune ViT-style backbone" than "train a CNN
  from scratch for a very long schedule," where constant LR is more common.
- `constant` is retained and documented as an available option — useful as a
  quick sanity-check ablation (e.g. "does the schedule choice itself change
  the selection-method ranking?") but not recommended as the default.
- `poly` (and any other schedule shape) is deliberately not reintroduced —
  two simple, well-understood options are enough to keep the "one frozen
  recipe" principle legible and auditable in the paper's methods section.

Warmup is expressed as **`warmup_epochs`**, resolved to steps the same way as
`epochs` (realized subset size × dataset's steps-per-epoch), so the warmup
*fraction* of the schedule stays constant across subset ratios even though
the absolute step counts shrink with the subset.

## Consequences

- **Config schema change:** `iterations`, `warmup_iters`, `warmup_ratio`
  (kept, unrelated), `poly_power`, `eval_every`, `save_every` are replaced by
  `epochs`, `warmup_epochs`, `eval_every_epochs`, `save_every_epochs` in every
  recipe YAML. `lr_schedule` now takes `constant` or `cosine` instead of `poly`.
- **`Trainer` is the single place epochs resolve to steps** — it computes
  `steps_per_epoch = len(train_ds) // batch_size` from the *realized* subset
  size at `__init__` time and derives `total_iters`/`warmup_iters`/
  `eval_every`/`save_every` from it. Nothing else in the codebase (selection,
  eval, scripts) reasons about iteration counts as a configured quantity —
  they remain a derived/logged artifact only (`runs.csv` still has an
  `iterations` column, now alongside a new `epochs` column, for provenance
  and GPU-hour analysis).
- **Recipe files renamed and split per-dataset:** `proxy_20k.yaml` (shared by
  Cityscapes/ADE20K) and `proxy_40k_ade20k.yaml`/`full_60k_cityscapes.yaml`/
  `proxy_10k_camvid.yaml` are replaced by `cityscapes_proxy.yaml`,
  `cityscapes_ceiling.yaml`, `ade20k_proxy.yaml`, `ade20k_ceiling.yaml`,
  `camvid_proxy.yaml`. A single recipe can no longer be shared across datasets
  of very different sizes, since the same epoch count now implies very
  different absolute compute per dataset.
- **The discriminativeness check (§4) must be re-run** under the new
  convention — it was validated (conceptually, not yet actually run) against
  iteration budgets, not epoch budgets, and the failure modes differ (see
  Open Question below).
- **§11's compute budget estimates in the plan doc are now conservative
  (likely overestimates)**, since they assumed every run costs the same
  regardless of ratio; under fixed-epoch, low-ratio runs finish
  proportionally faster. Not recalculated as part of this ADR — flagged for a
  follow-up pass once real run times are measured.

## Open question (carried forward, not resolved here)

At extreme low ratios (5%), fixed-epoch training could leave very few
absolute gradient steps (e.g. Cityscapes 5% ≈ 149 images → ~18 steps/epoch ×
50 epochs ≈ 900 steps total) — potentially too few to leave warmup and reach a
non-degenerate mIoU, which would look like "random collapses at 5%" for
reasons that are actually just undertraining, not a real coreset-selection
finding. The discriminativeness check (§4) is the intended detector for this;
if it fires at low ratios specifically, the fix is likely a **per-ratio epoch
floor** (train small subsets for more epochs than large ones, breaking strict
epoch-matching at the extreme end) rather than abandoning epoch-matching
everywhere. Not implemented pre-emptively — resolve only if the check
actually shows the problem.

## Amendment (2026-09-21, same day): fixed 100 epochs, not per-dataset calibration

The per-dataset calibrated table above (50/8/300 epochs for Cityscapes/ADE20K/CamVid,
chosen to match the old iteration budgets' wall-clock) was **overridden** by explicit
project decision: **every dataset now uses `epochs: 100`, unconditionally**, regardless of
the resulting compute cost. Simplicity of a single fixed number across every dataset was
judged more valuable than preserving the previous wall-clock envelope.

**Consequence, stated plainly:** this reintroduces exactly the compute blowup the
calibration was designed to avoid. ADE20K's full train set is ~7x Cityscapes', so 100
epochs there is ~252,600 steps (vs. ~20,000 before) — a rough estimate of **13-19 hours**
for a single full-data reference run, before any subset/method/seed runs. Cityscapes
grows more modestly (~37,100 steps, ~2.0-3.0h). The ceiling recipes were bumped
proportionally (Cityscapes 300, ADE20K 200 epochs) to stay meaningfully longer than the
now-larger proxy budgets, but `ade20k_ceiling.yaml` in particular (~505,200 steps) is a
many-hour-to-day-plus run and should only be launched once the proxy recipe's actual
measured wall-clock is in hand.

This is a known, accepted tradeoff, not an error — recorded here so the reasoning in the
original "Decision" section above (which argued for calibration) isn't mistaken for the
current state of the repo. The calibrated table is kept as a record of the alternative
that was available and explicitly not taken.

## Amendment 2 (2026-09-21, same day): poly restored as the default schedule; epoch counts reverted to per-dataset calibration

Amendment 1 (above) fixed every dataset to a uniform 100 epochs and switched the LR
schedule to `cosine`. Both are reverted here:

- **Schedule reverts to `poly` (power=1.0).** This is not a "fancy" schedule being
  swapped for a simpler one — `power=1.0` in `(1-progress)^power` collapses to a
  straight line, i.e. plain linear decay to zero, no more complex than `cosine`'s curve.
  The reason to prefer it specifically: it's the **literature-standard schedule for
  SegFormer** — the official SegFormer/mmseg fine-tuning config for ADE20K/Cityscapes
  uses exactly AdamW + `lr=6e-5` + poly(power=1.0) + short linear warmup, which is also
  where this project's `lr=6e-5` originates. `cosine` was adopted in the original
  Decision section above only because poly's standard status wasn't known at the time;
  once known, there's no remaining reason to deviate from it, since doing so would add
  an unforced difference from established practice for no benefit. `cosine` and
  `constant` remain implemented in `WarmupLR` as documented, available alternatives.
- **Epoch counts revert to the calibrated per-dataset table** from the original Decision
  section (Cityscapes 50/300 ceiling, ADE20K 8/16 ceiling, CamVid 300), undoing Amendment
  1's uniform-100 policy. The hardware-cost objection to uniform-100 (ADE20K ~13-19h per
  full-data run) was never resolved, only accepted temporarily; reverting removes that
  cost without giving up anything else, since the calibration's own reasoning (preserve
  the previously-considered-reasonable wall-clock envelope while still getting the
  fixed-epoch efficiency framing) still holds.

Net effect: this amendment returns the recipe to the original Decision section's numbers,
with the fixed-epoch (not fixed-iteration) convention as the one part of ADR 0001 that
was never in question and remains unchanged throughout.

## Alternatives considered

- **Keep fixed-iteration (status quo ante).** Rejected: doesn't match the
  paper's own real-world motivation or its closest comparable prior work;
  kept instead as a secondary ablation.
- **Fixed epochs with a literal epoch count of 100 everywhere (copying
  2505.01225 verbatim).** Originally rejected here on hardware-budget grounds
  (ADE20K's dataset is large enough that 100 epochs blows past the
  single-3090 budget, ~7x the calibrated ADE20K compute) in favor of
  per-dataset calibration — **then explicitly adopted anyway** by the
  2026-09-21 amendment above, which prioritized a single simple number over
  preserving the previous wall-clock envelope. The hardware-cost objection was
  not resolved, only accepted as a known tradeoff.
- **Poly decay retained instead of cosine.** Rejected per explicit request to
  limit to two simple, named options (`constant`, `cosine`); cosine was
  chosen over constant as the default for the reasons above, with constant
  kept as a documented fallback/ablation.
