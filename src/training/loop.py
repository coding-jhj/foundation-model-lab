from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from model.math_ops import cross_entropy


BatchProvider = Callable[[], tuple[torch.Tensor, torch.Tensor]]


def language_model_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token cross-entropy for a language model batch."""
    logits = model(input_ids)
    return cross_entropy(logits, targets)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    max_grad_norm: float | None = None,
) -> float:
    """Run one optimization step and return the detached scalar loss."""
    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive when provided")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = language_model_loss(model, input_ids, targets)
    loss.backward()

    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return float(loss.detach().cpu())


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_provider: BatchProvider,
    num_steps: int,
    max_grad_norm: float | None = None,
) -> list[float]:
    """Run a fixed number of language-model training steps."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    losses = []
    for _ in range(num_steps):
        input_ids, targets = batch_provider()
        losses.append(
            train_step(
                model,
                optimizer,
                input_ids,
                targets,
                max_grad_norm=max_grad_norm,
            )
        )

    return losses
