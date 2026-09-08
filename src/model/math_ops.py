from __future__ import annotations

from collections.abc import Callable

import torch


def softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute a numerically stable softmax along one dimension."""
    shifted_logits = logits - logits.amax(dim=dim, keepdim=True)
    exponentiated = shifted_logits.exp()
    return exponentiated / exponentiated.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean next-token cross-entropy for class-index targets."""
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits shape without the class dimension")

    log_probabilities = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    target_indices = targets.to(dtype=torch.long).unsqueeze(-1)
    target_log_probabilities = log_probabilities.gather(dim=-1, index=target_indices).squeeze(-1)
    return -target_log_probabilities.mean()


def adamw_update(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one functional AdamW update and return new state tensors."""
    if step < 1:
        raise ValueError("step must be at least 1")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("beta values must be in the interval [0, 1)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if parameter.shape != gradient.shape:
        raise ValueError("parameter and gradient must have the same shape")
    if parameter.shape != first_moment.shape or parameter.shape != second_moment.shape:
        raise ValueError("optimizer state tensors must match the parameter shape")

    updated_first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
    updated_second_moment = beta2 * second_moment + (1.0 - beta2) * gradient.square()

    first_bias_correction = 1.0 - beta1**step
    second_bias_correction = 1.0 - beta2**step
    corrected_first_moment = updated_first_moment / first_bias_correction
    corrected_second_moment = updated_second_moment / second_bias_correction

    adaptive_direction = corrected_first_moment / (corrected_second_moment.sqrt() + epsilon)
    update = adaptive_direction + weight_decay * parameter
    updated_parameter = parameter - learning_rate * update

    return updated_parameter, updated_first_moment, updated_second_moment


def numerical_gradient(
    function: Callable[[torch.Tensor], torch.Tensor],
    parameter: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Estimate a scalar function's gradient with central finite differences."""
    if not parameter.is_floating_point():
        raise TypeError("parameter must use a floating-point dtype")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    working_parameter = parameter.detach().clone()
    estimated_gradient = torch.zeros_like(working_parameter)
    flat_parameter = working_parameter.reshape(-1)
    flat_gradient = estimated_gradient.reshape(-1)

    for index in range(flat_parameter.numel()):
        original_value = flat_parameter[index].item()

        with torch.no_grad():
            flat_parameter[index] = original_value + epsilon
        positive_value = function(working_parameter)

        with torch.no_grad():
            flat_parameter[index] = original_value - epsilon
        negative_value = function(working_parameter)

        with torch.no_grad():
            flat_parameter[index] = original_value

        if positive_value.numel() != 1 or negative_value.numel() != 1:
            raise ValueError("function must return a scalar tensor")

        flat_gradient[index] = (
            positive_value.detach() - negative_value.detach()
        ) / (2.0 * epsilon)

    return estimated_gradient
