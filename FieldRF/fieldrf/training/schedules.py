"""Stateful optimizer-step learning-rate schedule for VAE adaptation."""

import math


class WarmupCosineSchedule:
    """Persist the learning-rate plan and completed optimizer steps on resume."""

    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=1e-7):
        if total_steps < 1 or warmup_steps < 0:
            raise ValueError("total_steps must be positive and warmup_steps nonnegative")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.global_step = 0
        self._apply()

    def _apply(self):
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if self.global_step < self.warmup_steps:
                factor = self.global_step / max(self.warmup_steps, 1)
            else:
                progress = min(max(
                    (self.global_step - self.warmup_steps)
                    / max(self.total_steps - self.warmup_steps, 1), 0.0,
                ), 1.0)
                floor = self.min_lr / base_lr
                factor = floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
            group["lr"] = base_lr * factor

    def step(self):
        self.global_step += 1
        self._apply()

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            "total_steps", "warmup_steps", "min_lr", "base_lrs", "global_step",
        )}

    def load_state_dict(self, state):
        if len(state["base_lrs"]) != len(self.optimizer.param_groups):
            raise ValueError("Checkpoint learning-rate groups differ from optimizer")
        for key in self.state_dict():
            setattr(self, key, state[key])
        self._apply()
