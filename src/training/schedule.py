from __future__ import annotations


def linear_warmup_scale(step: int, warmup_steps: int) -> float:
    """Return a linear learning-rate multiplier for a one-based training step."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if warmup_steps == 0:
        return 1.0

    return min(step / warmup_steps, 1.0)
