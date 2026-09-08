from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally restricting the count to trainable tensors."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
