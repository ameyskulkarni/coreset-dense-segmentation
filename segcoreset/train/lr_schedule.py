"""LR schedules (ADR 0001, docs/adr/0001-epoch-based-training.md). Three options:
`poly` (the SegFormer/mmseg literature-standard schedule and the current default),
`cosine`, and `constant` — all with a linear warmup. `Trainer` resolves
`epochs`/`warmup_epochs` to step counts before constructing this — the schedule itself
only ever deals in steps.
"""
from __future__ import annotations

import math

_KINDS = ("poly", "cosine", "constant")


class WarmupLR:
    """Linear warmup for `warmup_iters` steps (from `base_lr * warmup_ratio`), then decays
    to `min_lr` over the remaining steps: `poly` ((1-progress)**power, power=1.0 is a
    straight line — this is the SegFormer paper's own schedule), `cosine`, or holds
    `base_lr` flat (`constant`). Call `.step()` once per optimizer step; it sets
    `param_group['lr']` in place and returns the new LR."""

    def __init__(self, optimizer, base_lr: float, total_iters: int, warmup_iters: int,
                 kind: str = "poly", min_lr: float = 0.0, warmup_ratio: float = 1e-6,
                 power: float = 1.0):
        """Validate the schedule kind and immediately set the optimizer to the step-0 LR.

        Args:
            optimizer: Optimizer whose every param group's `lr` is overwritten.
            base_lr: Peak LR reached at the end of warmup.
            total_iters: Total optimizer steps in the run (warmup included).
            warmup_iters: Linear-warmup steps; 0 disables warmup.
            kind: `"poly"`, `"cosine"`, or `"constant"`.
            min_lr: LR at the end of decay (`poly`/`cosine` only).
            warmup_ratio: Warmup starts at `base_lr * warmup_ratio`.
            power: Exponent for `poly` decay (1.0 = linear).

        Raises:
            ValueError: If `kind` is not one of `_KINDS`.
        """
        if kind not in _KINDS:
            raise ValueError(f"lr_schedule must be one of {_KINDS}, got {kind!r}")
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.kind = kind
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.power = power
        self._step_count = 0
        self._set_lr(self._lr_at(0))

    def _lr_at(self, it: int) -> float:
        """Return the learning rate for (0-based) step `it`.

        Warmup interpolates linearly from `base_lr * warmup_ratio` toward `base_lr`. After
        warmup, `progress` runs 0 -> 1 over the remaining `total_iters - warmup_iters` steps
        (clamped), and the LR follows the configured decay toward `min_lr`.

        Args:
            it: Step index.
        """
        if self.warmup_iters > 0 and it < self.warmup_iters:
            alpha = it / max(1, self.warmup_iters)
            return self.base_lr * (self.warmup_ratio + (1 - self.warmup_ratio) * alpha)
        if self.kind == "constant":
            return self.base_lr
        progress = (it - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        if self.kind == "poly":
            return self.min_lr + (self.base_lr - self.min_lr) * (1 - progress) ** self.power
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float) -> None:
        """Write `lr` into every optimizer param group."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> float:
        """Advance the schedule by one step: set the LR for the current step, then increment.

        Call once per optimizer step (after `optimizer.step()`, as `Trainer` does).

        Returns:
            The LR that was just set (for logging).
        """
        lr = self._lr_at(self._step_count)
        self._set_lr(lr)
        self._step_count += 1
        return lr
