from __future__ import annotations

import torch


def sample_language_model_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input and next-token target windows from a one-dimensional token stream."""
    if tokens.ndim != 1:
        raise ValueError("tokens must be a one-dimensional tensor")
    if tokens.is_floating_point() or tokens.dtype == torch.bool:
        raise TypeError("tokens must use an integer dtype")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    if tokens.numel() <= context_length:
        raise ValueError("tokens must contain more values than context_length")

    number_of_start_positions = tokens.numel() - context_length
    starts = torch.randint(
        high=number_of_start_positions,
        size=(batch_size,),
        device=tokens.device,
    )
    offsets = torch.arange(context_length, device=tokens.device)
    windows = starts.unsqueeze(1) + offsets

    inputs = tokens[windows]
    targets = tokens[windows + 1]

    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)

    return inputs, targets
