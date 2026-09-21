"""نرخ هر گام تابعی روشن از شمارهٔ گام و یک افق ثابت است."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleConfig:
    mode: str = "constant"
    warmup_steps: int = 0
    decay_steps: int = 0
    min_lr_ratio: float = 0.1

    def __post_init__(self):
        if self.mode not in ("constant", "cosine"):
            raise ValueError("schedule must be constant or cosine")
        if any(type(n) is not int or n < 0 for n in (self.warmup_steps, self.decay_steps)):
            raise ValueError("warmup_steps and decay_steps must be nonnegative integers")
        if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0,1]")
        if self.mode == "cosine" and self.decay_steps <= self.warmup_steps:
            raise ValueError("cosine requires a fixed decay_steps greater than warmup_steps")
        if self.mode == "constant" and self.decay_steps:
            raise ValueError("constant schedule does not use decay_steps")

    def learning_rate(self, base_rate, step):
        """step از ۱ آغاز می‌شود و نرخِ به‌روزرسانی همان گام را می‌دهد."""
        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        if not math.isfinite(base_rate) or base_rate <= 0:
            raise ValueError("base learning rate must be finite and positive")
        if step <= self.warmup_steps:
            return base_rate * step / self.warmup_steps
        if self.mode == "constant":
            return base_rate
        fraction = min(1.0, (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps))
        factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
        return base_rate * factor
