from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit CPU/CUDA device or select CUDA automatically."""
    normalized = requested.strip().lower()

    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(normalized)
    except RuntimeError as error:
        raise ValueError(f"unsupported device: {requested}") from error

    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    return device
