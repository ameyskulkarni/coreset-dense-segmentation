"""Fixed-epoch training loop (§4, ADR 0001 — docs/adr/0001-epoch-based-training.md).

Every run trains for the SAME number of epochs over whatever subset it was given; a 10%
subset therefore does ~10% of the gradient steps of the full dataset (less total compute,
not more repetition) — the only variable across runs is data content, and the resulting
compute savings are themselves a reported outcome, not a confound to control away. The
recipe config speaks only in epochs (`epochs`, `warmup_epochs`, `eval_every_epochs`,
`save_every_epochs`); this file is the ONLY place that resolves epochs to step counts,
using the realized subset size (`len(self.train_ds)`), so nothing downstream needs to
reason about iterations directly. One `Trainer` serves every model in the registry (§3) —
architecture only changes which `build_model` returns, everything else (loss, schedule,
AMP, logging, eval) is shared.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.registry import build_dataset
from ..data.transforms import build_train_transform
from ..eval.evaluator import Evaluator
from ..models.registry import build_model
from ..utils.seed import set_seed
from .checkpoint import save_checkpoint
from .losses import build_loss
from .lr_schedule import WarmupLR

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int) -> None:
    """DataLoader workers are forked from the parent process and otherwise inherit its
    `random`/`numpy` RNG state verbatim, so all workers draw correlated augmentation
    parameters (RandomScale/RandomCrop/RandomHorizontalFlip in ../data/transforms.py use
    the plain `random` module). Reseed each worker from its own torch-assigned seed.

    Args:
        worker_id: Worker index (unused; the per-worker torch seed already differs).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Trainer:
    def __init__(self, cfg, subset_ids: list[str] | None, run_dir: Path, wandb_run=None):
        """Seed RNGs and build everything a run needs: data, model, loss, optimizer, schedule.

        This is the ONLY place `recipe.epochs` / `recipe.warmup_epochs` are converted to step
        counts, using the realized subset size: `steps_per_epoch = len(train_ds) // batch_size`.

        Args:
            cfg: Full resolved run config (needs `dataset`, `model`, `recipe`).
            subset_ids: Train image ids to use (from a subset file), or `None` for the full
                train split.
            run_dir: Directory for `last.pt` / `final.pt` checkpoints.
            wandb_run: Active W&B run to log to, or `None` to log only to stdout.

        Raises:
            ValueError: If the subset has fewer images than `recipe.batch_size` (zero steps
                per epoch).
        """
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.wandb_run = wandb_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(cfg.recipe.seed_train)

        train_tf = build_train_transform(cfg.dataset)
        self.train_ds = build_dataset(cfg.dataset, split=cfg.dataset.train_split, subset_ids=subset_ids, transform=train_tf)
        self.train_loader = self._infinite_loader(self.train_ds)

        # The ONLY place epochs -> steps conversion happens (ADR 0001). `drop_last=True`
        # on the DataLoader below means an epoch is `len(train_ds) // batch_size` steps;
        # matched here so `epochs` means what the config says regardless of subset size.
        self.steps_per_epoch = len(self.train_ds) // cfg.recipe.batch_size
        if self.steps_per_epoch == 0:
            raise ValueError(
                f"Subset of {len(self.train_ds)} images is smaller than batch_size "
                f"{cfg.recipe.batch_size} — can't complete even one step per epoch."
            )
        self.total_iters = cfg.recipe.epochs * self.steps_per_epoch
        self.warmup_iters = cfg.recipe.get("warmup_epochs", 0) * self.steps_per_epoch

        self.model = build_model(cfg.model, num_classes=cfg.dataset.num_classes).to(self.device)
        self.criterion = build_loss(cfg.dataset)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.recipe.lr, weight_decay=cfg.recipe.weight_decay)
        self.scheduler = WarmupLR(
            self.optimizer, base_lr=cfg.recipe.lr, total_iters=self.total_iters,
            warmup_iters=self.warmup_iters, kind=cfg.recipe.get("lr_schedule", "poly"),
            min_lr=cfg.recipe.get("min_lr", 0.0), warmup_ratio=cfg.recipe.get("warmup_ratio", 1e-6),
            power=cfg.recipe.get("poly_power", 1.0),
        )
        self.use_amp = cfg.recipe.precision == "amp_fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.evaluator = Evaluator(cfg.dataset, device=self.device)

    def _infinite_loader(self, dataset) -> Iterator[dict]:
        # multiprocessing_context="spawn": the model is already moved to CUDA in __init__
        # (before this loader's first batch is ever pulled), so by the time these workers
        # are actually created the main process already has a live CUDA context and its
        # background threads. Forking such a process (the Linux/PyTorch default) can
        # intermittently deadlock — if a background thread holds an internal lock at the
        # instant fork() runs, the forked worker inherits a permanently-locked copy with
        # no thread able to release it. "spawn" starts workers as fresh interpreters
        # instead of forking, avoiding this hazard entirely (standard fix for CUDA +
        # multiprocessing DataLoader workers). Costs a bit of extra worker startup time
        # (each re-imports the process); worth it for correctness on unattended runs.
        """Yield training batches forever, reshuffling on each pass over `dataset`.

        `drop_last=True`, so each pass is exactly `steps_per_epoch` batches. Worker processes
        (when `recipe.num_workers > 0`) are persistent, use the "spawn" start method (see the
        comment below), and are reseeded via `_seed_worker`.

        Args:
            dataset: The training dataset.

        Yields:
            Collated batch dicts with `"image"` `[B, 3, h, w]`, `"label"` `[B, h, w]`, and
            `"image_id"` (list of str).
        """
        loader = DataLoader(
            dataset, batch_size=self.cfg.recipe.batch_size, shuffle=True,
            num_workers=self.cfg.recipe.num_workers, drop_last=True, pin_memory=True,
            persistent_workers=self.cfg.recipe.num_workers > 0,
            worker_init_fn=_seed_worker if self.cfg.recipe.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.cfg.recipe.num_workers > 0 else None,
        )
        while True:
            yield from loader

    def _log(self, metrics: dict, step: int) -> None:
        """Log `metrics` to W&B at `step` if a run is attached; otherwise do nothing."""
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def train(self) -> dict:
        """Run the full training loop, then the final full-protocol evaluation.

        Per step: forward/backward under optional fp16 autocast with gradient scaling, optional
        gradient-norm clipping (`recipe.grad_clip_norm`), optimizer step, then LR schedule step.
        Every `recipe.log_every` steps, logs mean loss, LR, throughput, and fractional epoch.
        Every `recipe.eval_every_epochs` epochs (except at the very end), runs a cheap
        whole-image eval on `recipe.eval_max_images` (default 100) val images without
        boundary-F — a trend line, not comparable to the final Cityscapes metric. Every
        `recipe.save_every_epochs` epochs, overwrites `<run_dir>/last.pt`.

        After the loop, saves `<run_dir>/final.pt` and runs `Evaluator.evaluate` with the
        dataset's real protocol and boundary-F.

        Returns:
            A dict with `metrics` (final eval dict), `gpu_hours` (training wall-clock hours,
            excluding final eval), `epochs`, `iterations` (total steps), and `checkpoint`
            (path to `final.pt`).
        """
        cfg = self.cfg
        total_iters = self.total_iters
        eval_every = cfg.recipe.get("eval_every_epochs", 0) * self.steps_per_epoch
        save_every = cfg.recipe.get("save_every_epochs", 0) * self.steps_per_epoch
        log_every = cfg.recipe.get("log_every", 50)
        grad_clip = cfg.recipe.get("grad_clip_norm")

        logger.info(
            f"starting training: {cfg.recipe.epochs} epochs x {self.steps_per_epoch} steps/epoch "
            f"= {total_iters} total steps ({len(self.train_ds)} images, batch {cfg.recipe.batch_size})"
        )

        self.model.train()
        running_loss = 0.0
        window_t0 = time.time()
        start_time = time.time()

        for it in range(1, total_iters + 1):
            batch = next(self.train_loader)
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            lr = self.scheduler.step()
            running_loss += loss.item()

            if it % log_every == 0:
                avg_loss = running_loss / log_every
                running_loss = 0.0
                ips = log_every * cfg.recipe.batch_size / (time.time() - window_t0)
                window_t0 = time.time()
                epoch = it / self.steps_per_epoch
                self._log({"train/loss": avg_loss, "train/lr": lr, "train/imgs_per_sec": ips,
                            "train/epoch": epoch}, step=it)
                eta_min = (total_iters - it) * cfg.recipe.batch_size / ips / 60
                logger.info(
                    f"epoch {epoch:6.2f}/{cfg.recipe.epochs} step {it}/{total_iters} "
                    f"loss={avg_loss:.4f} lr={lr:.2e} imgs/s={ips:.1f} eta={eta_min:.0f}min"
                )

            if eval_every and it % eval_every == 0 and it < total_iters:
                metrics = self.evaluator.evaluate(
                    self.model, max_images=cfg.recipe.get("eval_max_images", 100), compute_boundary_f=False,
                    mode="whole_image", resize_short_side=cfg.recipe.get("eval_resize_short_side"),
                )
                self._log({f"val/{k}": v for k, v in metrics.items() if k != "per_class_iou"}, step=it)
                logger.info(f"[eval @ step {it}] val/miou={metrics['miou']:.4f} val/pixel_acc={metrics['pixel_acc']:.4f}")
                self.model.train()

            if save_every and it % save_every == 0:
                save_checkpoint(self.run_dir / "last.pt", self.model, self.optimizer, it)
                logger.info(f"checkpoint saved -> {self.run_dir / 'last.pt'}")

        gpu_hours = (time.time() - start_time) / 3600
        save_checkpoint(self.run_dir / "final.pt", self.model, self.optimizer, total_iters)

        logger.info("running final full-protocol evaluation...")
        final_metrics = self.evaluator.evaluate(self.model, compute_boundary_f=True)
        self._log({f"final/{k}": v for k, v in final_metrics.items() if k != "per_class_iou"}, step=total_iters)
        logger.info(
            f"training done in {gpu_hours:.2f} GPU-h — "
            f"miou={final_metrics['miou']:.4f} rare_class_miou={final_metrics.get('rare_class_miou')}"
        )

        return {
            "metrics": final_metrics,
            "gpu_hours": gpu_hours,
            "epochs": cfg.recipe.epochs,
            "iterations": total_iters,
            "checkpoint": str(self.run_dir / "final.pt"),
        }
