#!/usr/bin/env python
"""Append W&B runs that aren't in the auto-generated report yet, as new "## Experiment N"
sections in the same format as docs/results.md.

Writes to docs/results_wandb.md (created with an overview header on first run), never to
the hand-curated docs/results.md — the script refuses that path.

Every finished W&B run in the project is pulled via the public API. A run counts as already
reported if its W&B run id appears anywhere in the output file (e.g. in a `.../runs/<id>`
link), so re-running only adds what's new and never touches existing text. Unfinished runs
(running/crashed/failed) are skipped and get picked up on a later invocation once finished.

Runs are grouped into one section per experiment *family*: the run name with its ratio
(`_r0.2`), seed (`_s1`), lr (`_lr1e-4`) and trailing config-hash tokens removed, e.g.
`cs_random_r0.2_s1` -> `cs_random`, `cs_lrsweep_lr3e-5_full_s0` -> `cs_lrsweep_full`.
Within a section, the summary table has one row per (ratio, lr, epochs), aggregated over
*all* finished runs with that key (already-reported ones included, so mean ± std stays
correct when a ratio's seeds land across several invocations); the per-run table lists only
the new runs. Each section gets a Goal/Design description from `describe_family` (known
experiment families; unknown ones get a TODO) and, for selection methods other than random,
a comparison against the random baseline's ±1 std band at the same ratio (plan §7's "win"
criterion). Observations are left as a TODO to fill in by hand.

Metrics are the `final/*` keys of the run summary (full eval protocol). `n_img` and train
GPU-h come from `results/metrics/runs.csv` (matched on wandb_run_id) when available, since
W&B doesn't log them; otherwise n_img is "–" and GPU-h falls back to W&B runtime (marked †,
includes final eval).

Also regenerates docs/method_comparison.md in full on every run: every selection method vs
the random baseline at each ratio (absolute mIoU / rare-class mIoU, Δ with ±1 std band
verdicts, and per-class Δ IoU at --class-ratio from runs.csv). It holds no hand-written text.

Usage:
    python scripts/05_wandb_report.py                      # append new runs to docs/results_wandb.md
    python scripts/05_wandb_report.py --dry-run            # print the new sections instead
    python scripts/05_wandb_report.py --include '^cs_proto' --exclude 'disccheck|lrsweep'
    python scripts/05_wandb_report.py --out docs/results_proto.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PROJECT = "coreset-dense-segmentation"
DEFAULT_OUT = Path("docs/results_wandb.md")
PROTECTED = Path("docs/results.md")  # hand-curated; this script must never write it
DEFAULT_RUNS_CSV = Path("results/metrics/runs.csv")
DEFAULT_OVERVIEW = Path("docs/method_comparison.md")

# Name tokens that vary *within* an experiment family: ratio, seed, lr, config hash.
_VARYING_TOKEN = re.compile(r"^(r\d+(\.\d+)?|s\d+|lr[\d.eE+-]+|[0-9a-f]{12})$")
_RUN_ID_IN_MD = re.compile(r"/runs/([A-Za-z0-9]+)")
_EXPERIMENT_HEADING = re.compile(r"^## Experiment (\d+)\b", re.MULTILINE)

NEW_FILE_HEADER = """# Experiment Results Log (auto-generated from W&B)

Running record of every finished W&B run in the project, appended by
`scripts/05_wandb_report.py`. Each invocation adds only runs whose W&B id isn't in this file
yet, as new `## Experiment N` sections; existing text is never rewritten, so Goal/Observation
notes added by hand are safe. The hand-curated narrative log is `docs/results.md` (separate,
not touched by the script).

## About these experiments

The project asks whether **label-free coreset selection** methods that beat random for
image classification (prototypicality, embedding coverage, ...) also beat random for
**semantic segmentation**, and in particular for **rare classes** — the hypothesis
(`segmentation-coreset-experiment-plan.md` §1) is that they don't, because image-level
selection can't isolate the small rare-class regions that dense prediction depends on.

Every run trains the same model with the same frozen recipe; the only thing that changes
between methods is *which images* are in the training subset:

- **Model:** SegFormer-B0 (`nvidia/mit-b0` ImageNet init). Cityscapes: 2975 train / 500 val,
  19 classes, 512x1024 crops.
- **Recipe** (`configs/recipe/cityscapes_proxy.yaml`): **epoch-based**, 100 epochs, batch 8,
  AdamW lr 1e-4 (chosen by the LR sweep), wd 0.01, poly decay to 0, 6 warmup epochs, fp16.
  Steps scale with subset size (ADR 0001): a 20% subset gets ~20% of the full run's
  gradient steps, so compute savings are part of the result rather than held fixed.
- **Seeds:** for subset runs, seed N fixes both the subset draw and the training RNG; at
  ratio 1.0 it only varies training stochasticity. 3 seeds per point unless stated.
- **Experiment ladder so far:** (1) recipe calibration — LR sweep on full data, then a
  discriminativeness check that full data clearly beats a 10% random subset; (2) the
  random-selection baseline across ratios 1.0 → 0.05, the variance band every method is
  judged against; (3) selection methods (prototypicality first) on the same ratio ladder.

## Conventions

- All metrics are **final eval** numbers (W&B `final/*`), in **%**. Cityscapes uses the
  sliding-window protocol (512x1024 tiles, stride 341x683, all 500 val images). Mid-training
  `val/*` curves in W&B are a cheap whole-image 100-image trend line and are *not* comparable.
- **Rare-class mIoU** = mean IoU over the frozen bottom-K classes by train-set pixel
  frequency (`results/rare_classes/<dataset>.json`; K = 5 for Cityscapes) — the thesis metric.
- `mean ± std` is across runs, **sample std (ddof=1)**; the Seeds column shows which
  training seeds went into each row.
- **Retention** = mean mIoU(subset) / mean mIoU(full-100%, same dataset/model/recipe/lr/epochs).
- A method **beats random** at a ratio only if its mean is outside random's ±1 std band
  there (plan §7); the "vs random" tables mark this per metric.
- n_img and train GPU-h come from `results/metrics/runs.csv` when the run has a row there
  (W&B doesn't log them); † marks a fallback to W&B runtime, which includes final eval.

---
"""


def describe_family(family: str, runs: list[Run]) -> str:
    """Goal/design paragraph for a known experiment family (TODO placeholder otherwise)."""
    if "lrsweep" in family:
        return ("**Goal.** Recipe calibration (plan §4, §9): pick the learning rate for the frozen "
                "proxy recipe by training on 100% of the data, one seed per LR. The winner becomes "
                "`recipe.lr` for every later run; these are calibration runs, not method results. "
                "Repeated runs at the same LR measure the reproducibility noise floor.")
    if "disccheck" in family:
        return ("**Goal.** Recipe calibration (plan §9): a *discriminativeness check* — the recipe is "
                "only useful for comparing subsets if full data clearly beats a small random subset. "
                "Trains 10% random subsets at each candidate LR and compares with the full-data run at "
                "the same LR. Note: the `_s1/_s2` name suffixes refer to the subset file, but these runs "
                "all used training seed 0 (see Seeds column).")
    method = runs[0].method
    if method == "random" or (method == "full" and "random" in family):
        return ("**Goal.** Random-selection baseline (plan §5.1, §7–8): uniform random subsets without "
                "replacement across the ratio ladder, with ratio 1.0 as the full-data reference "
                "(retention denominator). Its per-ratio ±1 std band is what every selection method must "
                "clear to count as a win.")
    m = re.search(r"proto-(hard|easy)-k(\d+)-bal(\d+)-(.+)$", method)
    if m:
        keep, k, bal, feats = m.groups()
        keep_txt = ("the *hardest* (most atypical) images, farthest from their nearest prototype"
                    if keep == "hard" else
                    "the *easiest* (most prototypical) images, closest to their nearest prototype")
        return ("**Goal.** Prototypicality selection (Sorscher et al. 2022 §6; `docs/prototypicality.md`), "
                "the canonical label-free \"representativeness\" selector. Embeds every training image "
                f"(`{feats}`), clusters the embeddings with spherical k-means (k = {k}), scores each image "
                f"by cosine distance to its nearest centroid, and keeps {keep_txt}, with cluster balance "
                f"{int(bal) / 100:g} so the subset can't collapse onto a few scene types. The question is "
                "whether keeping atypical *images* helps segmentation, and rare classes in particular, "
                "beyond random at the same ratio.")
    return f"**Goal.** _TODO_ (selection method `{method}`)."


@dataclass
class Run:
    """One finished W&B run, flattened to the fields the report uses."""

    id: str
    name: str
    url: str
    created_at: str
    family: str
    dataset: str
    model: str
    method: str
    ratio: float
    seed: int | None
    lr: float
    epochs: int
    recipe: str
    steps: int | None
    miou: float
    rare_miou: float | None
    pixel_acc: float | None
    boundary_f: float | None
    runtime_h: float | None
    config: dict

    @property
    def key(self) -> tuple[float, float, int]:
        """Summary-table row key within a family."""
        return (self.ratio, self.lr, self.epochs)

    @property
    def ref_key(self) -> tuple:
        """What a full-data run must share with this run to be its retention reference."""
        return (self.dataset, self.model, self.recipe, self.lr, self.epochs)


def family_of(name: str) -> str:
    """Strip ratio/seed/lr/hash tokens from a run name, e.g. `cs_random_r0.2_s1` -> `cs_random`."""
    kept = [t for t in name.split("_") if not _VARYING_TOKEN.match(t)]
    return "_".join(kept) or name


def fetch_runs(entity: str | None, project: str) -> tuple[list[Run], list[tuple[str, str, str]]]:
    """Pull every run in the project; return (finished runs with final metrics, skipped).

    Skipped entries are (name, id, reason) for runs that aren't finished or lack `final/miou`.
    """
    import wandb

    api = wandb.Api(timeout=60)
    path = f"{entity or api.default_entity}/{project}"
    finished, skipped = [], []
    for r in api.runs(path, order="+created_at"):
        s = r.summary
        if r.state != "finished":
            skipped.append((r.name, r.id, r.state))
            continue
        if s.get("final/miou") is None:
            skipped.append((r.name, r.id, "no final/miou in summary"))
            continue
        cfg = dict(r.config)
        recipe = cfg.get("recipe", {}) or {}
        runtime = s.get("_runtime")
        finished.append(Run(
            id=r.id, name=r.name, url=r.url, created_at=r.created_at,
            family=family_of(r.name),
            dataset=(cfg.get("dataset") or {}).get("name", "?"),
            model=(cfg.get("model") or {}).get("name", "?"),
            method=cfg.get("selection_method", "?"),
            ratio=float(cfg.get("ratio") if cfg.get("ratio") is not None else float("nan")),
            seed=recipe.get("seed_train"),
            lr=float(recipe.get("lr", float("nan"))),
            epochs=int(recipe.get("epochs", -1)),
            recipe=recipe.get("name", "?"),
            steps=s.get("_step"),
            miou=s["final/miou"],
            rare_miou=s.get("final/rare_class_miou"),
            pixel_acc=s.get("final/pixel_acc"),
            boundary_f=s.get("final/boundary_f"),
            runtime_h=runtime / 3600 if runtime is not None else None,
            config=cfg,
        ))
    return finished, skipped


def load_ledger(path: Path) -> dict[str, dict]:
    """runs.csv rows keyed by wandb_run_id (empty if the ledger doesn't exist)."""
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["wandb_run_id"]: row for row in csv.DictReader(f) if row.get("wandb_run_id")}


# ---------------------------------------------------------------------------------- formatting

def pct(x: float | None) -> str:
    """Fraction -> percent with 2 decimals, or "–" if missing."""
    return "–" if x is None else f"{100 * x:.2f}"


def mean_std(values: list[float | None], scale: float = 100.0, digits: int = 2) -> str:
    """`mean ± sample std` of the non-missing values (just the mean when n == 1)."""
    vals = [v * scale for v in values if v is not None]
    if not vals:
        return "–"
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {statistics.stdev(vals):.{digits}f}"


def fmt_lr(lr: float) -> str:
    """1e-04 -> 1e-4, 6e-05 -> 6e-5."""
    mant, exp = f"{lr:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def fmt_ratio(r: float) -> str:
    return "1.0 (full)" if r == 1.0 else f"{r:g}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ sections

def pick_reference(family_runs: list[Run], all_runs: list[Run], ref_key: tuple) -> tuple[list[Run], str | None]:
    """Full-data runs to use as the retention denominator for `ref_key`.

    Prefers the family's own ratio-1.0 runs; otherwise the family with the most matching
    full-data runs (so a single calibration run doesn't get pooled with a 3-seed baseline).
    """
    def is_ref(r: Run) -> bool:
        return r.method == "full" and r.ratio == 1.0 and r.ref_key == ref_key

    own = [r for r in family_runs if is_ref(r)]
    if own:
        return own, own[0].family
    by_family = defaultdict(list)
    for r in all_runs:
        if is_ref(r):
            by_family[r.family].append(r)
    if not by_family:
        return [], None
    fam = max(by_family, key=lambda f: (len(by_family[f]), max(r.created_at for r in by_family[f])))
    return by_family[fam], fam


def random_band_table(new_keys: list[tuple], by_key: dict, all_runs: list[Run]) -> str | None:
    """Delta-vs-random table for a non-random method; None if no matching random runs exist.

    Random runs are matched on ratio + ref_key (dataset/model/recipe/lr/epochs), taken from
    the random family with the most such runs (so calibration runs don't get pooled in).
    """
    pools = defaultdict(lambda: defaultdict(list))  # family -> key -> runs
    for r in all_runs:
        if r.method == "random":
            pools[r.family][(r.ratio,) + r.ref_key].append(r)
    pools = {f: dict(keys) for f, keys in pools.items()}  # plain dicts: .get must not insert
    if not pools:
        return None

    def verdict(vals: list[float], rnd: list[float]) -> str:
        m, rm = statistics.mean(vals), statistics.mean(rnd)
        if len(rnd) < 2:
            return f"{100 * (m - rm):+.2f}"
        sd = statistics.stdev(rnd)
        tag = "above band" if m > rm + sd else "below band" if m < rm - sd else "inside band"
        return f"{100 * (m - rm):+.2f} ({tag})"

    rows = []
    for key in new_keys:
        runs = by_key[key]
        match = (key[0],) + runs[0].ref_key
        # Most matching runs; ties go to the most recent family (the baseline, not calibration).
        fam = max(pools, key=lambda f: (len(pools[f].get(match, [])),
                                        max((r.created_at for r in pools[f].get(match, [])), default="")))
        rnd = pools[fam].get(match, [])
        if not rnd:
            continue
        rare = [r.rare_miou for r in runs if r.rare_miou is not None]
        rnd_rare = [r.rare_miou for r in rnd if r.rare_miou is not None]
        rows.append([
            f"{key[0]:g}", f"`{fam}` ({len(rnd)})",
            mean_std([r.miou for r in rnd]), verdict([r.miou for r in runs], [r.miou for r in rnd]),
            mean_std(rnd_rare), verdict(rare, rnd_rare) if rare and rnd_rare else "–",
        ])
    if not rows:
        return None
    return table(["Ratio", "Random runs (n)", "Random mIoU (%)", "Δ mIoU (pts)",
                  "Random rare mIoU (%)", "Δ rare mIoU (pts)"], rows)


def setup_block(runs: list[Run]) -> str:
    """Setup bullets from the family's configs; fields that differ across runs are listed as sets."""
    def vals(fn) -> str:
        seen = []
        for r in runs:
            v = fn(r)
            if v not in seen:
                seen.append(v)
        return " / ".join(str(v) for v in seen)

    rc = lambda r, k: (r.config.get("recipe") or {}).get(k)  # noqa: E731
    ds = lambda r, k: (r.config.get("dataset") or {}).get(k)  # noqa: E731
    lines = [
        f"- Model: {vals(lambda r: r.model)} (`{vals(lambda r: (r.config.get('model') or {}).get('pretrained'))}` init); "
        f"dataset {vals(lambda r: r.dataset)}, {vals(lambda r: ds(r, 'num_classes'))} classes, "
        f"crop {vals(lambda r: 'x'.join(map(str, ds(r, 'crop_size') or [])))}.",
        f"- Recipe `{vals(lambda r: r.recipe)}`: {vals(lambda r: r.epochs)} epochs, batch {vals(lambda r: rc(r, 'batch_size'))}, "
        f"{vals(lambda r: rc(r, 'optimizer'))}, lr {vals(lambda r: fmt_lr(r.lr))}, wd {vals(lambda r: rc(r, 'weight_decay'))}, "
        f"{vals(lambda r: rc(r, 'lr_schedule'))} schedule, {vals(lambda r: rc(r, 'warmup_epochs'))} warmup epochs, "
        f"{vals(lambda r: rc(r, 'precision'))}.",
        f"- Selection: `{vals(lambda r: r.method)}`; ratios {', '.join(f'{x:g}' for x in sorted({r.ratio for r in runs}, reverse=True))}; "
        f"training seeds {', '.join(str(s) for s in sorted({r.seed for r in runs if r.seed is not None}))}.",
        f"- Eval: {vals(lambda r: (ds(r, 'eval') or {}).get('mode'))}"
        + (f" (stride {vals(lambda r: 'x'.join(map(str, (ds(r, 'eval') or {}).get('stride') or [])))})"
           if any((ds(r, 'eval') or {}).get('stride') for r in runs) else "")
        + f"; rare-class mIoU over bottom-{vals(lambda r: ds(r, 'rare_class_k'))} classes.",
        f"- Code: git {vals(lambda r: r.config.get('git_commit', '?')[:7])}.",
    ]
    return "\n".join(lines)


def build_section(n: int, family: str, new: list[Run], family_all: list[Run], all_runs: list[Run],
                  ledger: dict[str, dict]) -> str:
    """Markdown for one "## Experiment N" section covering the `new` runs of `family`."""
    new_keys = sorted({r.key for r in new}, key=lambda k: (-k[0], k[1], k[2]))
    by_key = defaultdict(list)
    for r in family_all:
        by_key[r.key].append(r)

    def n_img(r: Run) -> str:
        return ledger.get(r.id, {}).get("n_images") or "–"

    def gpu_h(r: Run) -> float | None:
        v = ledger.get(r.id, {}).get("gpu_hours_train")
        return float(v) if v else None

    def gpu_h_str(r: Run) -> str:
        if gpu_h(r) is not None:
            return f"{gpu_h(r):.2f}"
        return f"{r.runtime_h:.2f}†" if r.runtime_h is not None else "–"

    refs_used, summary_rows, any_dagger = {}, [], False
    for key in new_keys:
        runs = by_key[key]
        ref_runs, ref_fam = pick_reference(family_all, all_runs, runs[0].ref_key)
        if ref_runs:
            refs_used[ref_fam] = ref_runs
            retention = f"{100 * statistics.mean(r.miou for r in runs) / statistics.mean(r.miou for r in ref_runs):.1f}%"
        else:
            retention = "–"
        hours = [gpu_h(r) for r in runs]
        if all(h is not None for h in hours):
            hours_str = mean_std(hours, scale=1.0)
        else:
            hours_str = mean_std([r.runtime_h for r in runs], scale=1.0) + "†"
            any_dagger = True
        summary_rows.append([
            fmt_ratio(key[0]), fmt_lr(key[1]), str(key[2]),
            ",".join(str(r.seed) for r in sorted(runs, key=lambda r: (r.seed is None, r.seed))),
            " / ".join(sorted({n_img(r) for r in runs})),
            " / ".join(str(s) for s in sorted({r.steps for r in runs if r.steps is not None})) or "–",
            mean_std([r.miou for r in runs]), mean_std([r.rare_miou for r in runs]),
            mean_std([r.pixel_acc for r in runs]), mean_std([r.boundary_f for r in runs]),
            hours_str, retention,
        ])

    per_run_rows = []
    for r in sorted(new, key=lambda r: (-r.ratio, r.lr, r.epochs, r.seed if r.seed is not None else -1, r.created_at)):
        any_dagger |= gpu_h(r) is None and r.runtime_h is not None
        per_run_rows.append([
            r.name, f"{r.ratio:g}", fmt_lr(r.lr), str(r.seed), n_img(r),
            pct(r.miou), pct(r.rare_miou), pct(r.pixel_acc), pct(r.boundary_f), gpu_h_str(r),
            f"[{r.id}]({r.url})",
        ])

    datasets = sorted({r.dataset for r in new})
    models = sorted({r.model for r in new})
    already = len(family_all) - len(new)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out = [
        f"## Experiment {n} — `{family}`, {'/'.join(datasets)} / {'/'.join(models)}",
        "",
        f"<!-- generated by scripts/05_wandb_report.py on {stamp}: {len(new)} new run(s)"
        + (f"; {already} earlier run(s) of this family already reported above" if already else "") + " -->",
        "",
        describe_family(family, family_all),
        "",
        "**Setup** (from W&B run configs)",
        setup_block(new),
        "",
        "### Summary (mean ± std over runs)",
        "",
        table(["Ratio", "LR", "Epochs", "Seeds", "n_img", "Steps", "mIoU (%)", "Rare-class mIoU (%)",
               "Pixel acc (%)", "Boundary F (%)", "Train GPU-h", "Retention (mIoU)"], summary_rows),
        "",
    ]
    notes = []
    if already:
        notes.append("Summary rows aggregate every finished run with that ratio/lr/epochs in this family, "
                     "including runs already listed in earlier sections; the per-run table lists only new runs.")
    for fam, ref_runs in refs_used.items():
        notes.append(f"Retention denominator: `{fam}` full-data runs "
                     f"({', '.join(r.name for r in ref_runs)}), mean mIoU {pct(statistics.mean(r.miou for r in ref_runs))}.")
    if any_dagger:
        notes.append("† no `runs.csv` row for the run: W&B runtime, which also includes final eval.")
    dup_seeds = [k for k in new_keys if len({r.seed for r in by_key[k]}) < len(by_key[k])]
    if dup_seeds:
        notes.append("Repeated training seed within a row (" + "; ".join(
            f"ratio {k[0]:g} lr {fmt_lr(k[1])}" for k in dup_seeds) + "): the spread is not across seeds.")
    out += [f"- {n}" for n in notes]
    if not any(r.method in ("random", "full") for r in new):
        band = random_band_table(new_keys, by_key, all_runs)
        if band:
            out += ["", "### vs random baseline (same ratio, recipe, lr)", "",
                    "Δ = method mean − random mean; band = random mean ± 1 sample std.", "", band]
    out += [
        "",
        "Observations",
        "- _TODO_",
        "",
        "### Per-run results",
        "",
        table(["Run", "Ratio", "LR", "Seed", "n_img", "mIoU", "Rare mIoU", "Pixel acc", "Boundary F",
               "GPU-h", "W&B"], per_run_rows),
        "",
        "---",
        "",
    ]
    return "\n".join(out)


def class_names_for(dataset: str) -> list[str] | None:
    """Class names from the dataset class, if it defines them statically (Cityscapes, CamVid)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from segcoreset.data.registry import DATASET_REGISTRY
        return list(getattr(DATASET_REGISTRY[dataset], "class_names", None) or []) or None
    except Exception:  # noqa: BLE001 — names are cosmetic; fall back to ids
        return None


def build_overview(all_runs: list[Run], ledger: dict[str, dict], class_ratio: float) -> str:
    """Cross-method comparison page: every selection method vs random, one table per metric.

    Regenerated in full on every invocation (it holds no hand-written text). One block per
    (dataset, model). The random baseline is the random family with the most runs, so
    calibration runs (`disccheck`) never serve as the baseline.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = [
        "# Method comparison vs random (auto-generated)",
        "",
        f"Regenerated in full by `scripts/05_wandb_report.py` on {stamp}; do not edit by hand (notes "
        "belong in `docs/results_wandb.md`). Final-eval metrics in %, mean over seeds.",
        "",
        "**How to read:** Δ = method mean − random mean at the same ratio, in points. "
        "**▲ / ▼ = outside random's ±1 std band** (the plan §7 bar for a win / a loss); unmarked = "
        "inside the band, i.e. not distinguishable from random. Each cell is 3 seeds unless marked (n=…). "
        "Methods are the selection presets; `cls` / `patchmean` is the embedding the selector used.",
        "",
    ]
    by_dm = defaultdict(list)
    for r in all_runs:
        by_dm[(r.dataset, r.model)].append(r)

    for (dataset, model), runs in sorted(by_dm.items()):
        randoms = defaultdict(list)
        for r in runs:
            if r.method == "random":
                randoms[r.family].append(r)
        methods = sorted({r.method for r in runs if r.method not in ("random", "full")})
        if not randoms or not methods:
            continue
        base_fam = max(randoms, key=lambda f: (len(randoms[f]), max(r.created_at for r in randoms[f])))
        base = randoms[base_fam]
        ratios = sorted({r.ratio for r in base}, reverse=True)

        def cell_runs(method: str, ratio: float) -> list[Run]:
            return [r for r in runs if r.method == method and r.ratio == ratio]

        def base_runs(m_runs: list[Run], ratio: float) -> list[Run]:
            keys = {r.ref_key for r in m_runs}
            return [r for r in base if r.ratio == ratio and r.ref_key in keys]

        def n_tag(rs: list) -> str:
            return "" if len(rs) == 3 else f" (n={len(rs)})"

        out += [f"## {dataset} / {model}", "",
                f"Random baseline: `{base_fam}` ({len(base)} runs). Methods: {len(methods)}.", ""]

        # Absolute mIoU / rare mIoU.
        for metric, label in (("miou", "mIoU"), ("rare_miou", "Rare-class mIoU")):
            rows = []
            for name in ["random"] + methods:
                cells = []
                for ratio in ratios:
                    rs = [r for r in base if r.ratio == ratio] if name == "random" else cell_runs(name, ratio)
                    vals = [getattr(r, metric) for r in rs]
                    cells.append(mean_std(vals) + n_tag(rs) if rs else "–")
                rows.append([f"**{name}**" if name == "random" else f"`{name}`"] + cells)
            out += [f"### {label} (%), mean ± std", "",
                    table(["Method"] + [fmt_ratio(x) for x in ratios], rows), ""]

        # Deltas vs random.
        for metric, label in (("miou", "mIoU"), ("rare_miou", "rare-class mIoU")):
            rows = []
            for name in methods:
                cells, deltas, wins, losses, tested = [], [], 0, 0, 0
                for ratio in ratios:
                    rs = cell_runs(name, ratio)
                    bs = base_runs(rs, ratio)
                    m_vals = [getattr(r, metric) for r in rs if getattr(r, metric) is not None]
                    b_vals = [getattr(r, metric) for r in bs if getattr(r, metric) is not None]
                    if not m_vals or not b_vals:
                        cells.append("–")
                        continue
                    d = 100 * (statistics.mean(m_vals) - statistics.mean(b_vals))
                    sd = 100 * statistics.stdev(b_vals) if len(b_vals) > 1 else 0.0
                    mark = " ▲" if d > sd and sd else " ▼" if d < -sd and sd else ""
                    wins += mark == " ▲"
                    losses += mark == " ▼"
                    tested += 1
                    deltas.append(d)
                    cells.append(f"{d:+.2f}{mark}{n_tag(rs)}")
                summary = f"{wins}▲ {losses}▼ / {tested}" if tested else "–"
                avg = f"{statistics.mean(deltas):+.2f}" if deltas else "–"
                rows.append([f"`{name}`"] + cells + [avg, summary])
            out += [f"### Δ {label} vs random (points)", "",
                    table(["Method"] + [fmt_ratio(x) for x in ratios] + ["Mean Δ", "Outside band"], rows), ""]

        # Per-class deltas at one ratio (needs per-class IoU from runs.csv).
        names = class_names_for(dataset)

        def per_class(rs: list[Run]) -> dict[int, float] | None:
            maps = [json.loads(ledger[r.id]["per_class_iou_json"]) for r in rs
                    if r.id in ledger and ledger[r.id].get("per_class_iou_json")]
            if not maps:
                return None
            return {int(c): statistics.mean(m[c] for m in maps) for c in maps[0]}

        base_pc = per_class([r for r in base if r.ratio == class_ratio])
        method_pc = {m: per_class(cell_runs(m, class_ratio)) for m in methods}
        method_pc = {m: pc for m, pc in method_pc.items() if pc}
        if base_pc and method_pc:
            rare_ids = []
            cache = (runs[0].config.get("dataset") or {}).get("rare_classes_cache")
            if cache and Path(cache).exists():
                rare_ids = json.loads(Path(cache).read_text())["rare_class_ids"]
            # Rare classes, plus any class that moves >= 2 pts for some method.
            movers = [c for c in sorted(base_pc) if c not in rare_ids
                      and any(abs(pc[c] - base_pc[c]) >= 0.02 for pc in method_pc.values())]
            cols = rare_ids + movers
            label = lambda c: (names[c] if names and c < len(names) else f"class {c}") + (" ★" if c in rare_ids else "")  # noqa: E731
            rows = [["**random IoU**"] + [f"{100 * base_pc[c]:.1f}" for c in cols]]
            rows += [[f"`{m}`"] + [f"{100 * (pc[c] - base_pc[c]):+.1f}" for c in cols] for m, pc in method_pc.items()]
            out += [f"### Per-class Δ IoU vs random at ratio {class_ratio:g} (points)", "",
                    "★ = frozen rare class (counted in rare-class mIoU). Other columns: classes that move by "
                    "≥ 2 points for at least one method. First row is random's absolute IoU for scale.", "",
                    table(["Method"] + [label(c) for c in cols], rows), ""]
    return "\n".join(out)


def main():
    """CLI entry point: fetch W&B runs, append sections for the unreported ones."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"markdown report to append to (default {DEFAULT_OUT})")
    parser.add_argument("--entity", default=None, help="W&B entity (default: your API key's default entity)")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV, help="ledger used for n_img / train GPU-h")
    parser.add_argument("--include", default=None, help="only report runs whose name matches this regex")
    parser.add_argument("--exclude", default=None, help="skip runs whose name matches this regex")
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW,
                        help=f"cross-method comparison page, regenerated in full each run (default {DEFAULT_OVERVIEW})")
    parser.add_argument("--class-ratio", type=float, default=0.2, help="ratio for the overview's per-class table")
    parser.add_argument("--dry-run", action="store_true", help="print the new sections instead of writing")
    args = parser.parse_args()
    for path in (args.out, args.overview):
        if path.resolve() == PROTECTED.resolve():
            parser.error(f"refusing to write {PROTECTED} (hand-curated); pick another path")

    text = args.out.read_text() if args.out.exists() else NEW_FILE_HEADER
    seen_ids = set(_RUN_ID_IN_MD.findall(text))
    next_n = max((int(m) for m in _EXPERIMENT_HEADING.findall(text)), default=0) + 1

    all_runs, skipped = fetch_runs(args.entity, args.project)
    ledger = load_ledger(args.runs_csv)

    new = [r for r in all_runs if r.id not in seen_ids
           and (args.include is None or re.search(args.include, r.name))
           and (args.exclude is None or not re.search(args.exclude, r.name))]

    for name, rid, reason in skipped:
        if rid not in seen_ids:
            print(f"skipped {name} ({rid}): {reason}", file=sys.stderr)

    overview = build_overview(all_runs, ledger, args.class_ratio)
    if not args.dry_run:
        args.overview.parent.mkdir(parents=True, exist_ok=True)
        args.overview.write_text(overview)
        print(f"Regenerated {args.overview}", file=sys.stderr)
    if not new:
        print(f"No new finished runs ({len(all_runs)} finished in W&B, {len(seen_ids)} ids already in {args.out}).",
              file=sys.stderr)
        return

    families = defaultdict(list)
    for r in all_runs:
        families[r.family].append(r)
    new_by_family = defaultdict(list)
    for r in new:
        new_by_family[r.family].append(r)

    # Sections in order of each family's first new run.
    order = sorted(new_by_family, key=lambda f: min(r.created_at for r in new_by_family[f]))
    sections = [build_section(next_n + i, fam, new_by_family[fam], families[fam], all_runs, ledger)
                for i, fam in enumerate(order)]

    if args.dry_run:
        print("\n".join(sections))
    else:
        sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + sep + "\n".join(sections))
    print(f"{'Would add' if args.dry_run else 'Added'} {len(new)} run(s) in {len(sections)} section(s) "
          f"(Experiment {next_n}–{next_n + len(sections) - 1}) to {args.out}: "
          + ", ".join(f"{f} ({len(new_by_family[f])})" for f in order), file=sys.stderr)


if __name__ == "__main__":
    main()
