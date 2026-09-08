from pathlib import Path

import pytest
import torch


def _math_ops():
    try:
        from model.math_ops import adamw_update, cross_entropy, numerical_gradient, softmax
    except ModuleNotFoundError as error:
        pytest.fail(f"Expected model.math_ops to exist: {error}", pytrace=False)

    return adamw_update, cross_entropy, numerical_gradient, softmax


def test_softmax_is_normalized_and_numerically_stable() -> None:
    _, _, _, softmax = _math_ops()
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]])

    probabilities = softmax(logits, dim=-1)

    assert torch.isfinite(probabilities).all()
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2))


def test_cross_entropy_matches_negative_log_likelihood() -> None:
    _, cross_entropy, _, softmax = _math_ops()
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    targets = torch.tensor([0, 2])

    actual = cross_entropy(logits, targets)
    probabilities = softmax(logits, dim=-1)
    expected = -torch.log(probabilities[torch.arange(2), targets]).mean()

    assert torch.allclose(actual, expected, atol=1e-6)


def test_adamw_update_applies_bias_correction_and_weight_decay() -> None:
    adamw_update, _, _, _ = _math_ops()
    parameter = torch.tensor([1.0, -2.0])
    gradient = torch.tensor([0.1, -0.2])
    first_moment = torch.zeros_like(parameter)
    second_moment = torch.zeros_like(parameter)

    updated_parameter, updated_first, updated_second = adamw_update(
        parameter=parameter,
        gradient=gradient,
        first_moment=first_moment,
        second_moment=second_moment,
        step=1,
        learning_rate=0.01,
        weight_decay=0.1,
    )

    expected_first = 0.1 * gradient
    expected_second = 0.001 * gradient.square()
    expected_adam_direction = gradient / (expected_second.sqrt() / (1 - 0.999) ** 0.5 + 1e-8)
    expected_update = expected_adam_direction + 0.1 * parameter
    expected_parameter = parameter - 0.01 * expected_update

    assert torch.allclose(updated_first, expected_first)
    assert torch.allclose(updated_second, expected_second)
    assert torch.allclose(updated_parameter, expected_parameter, atol=1e-6)


def test_numerical_gradient_matches_autograd() -> None:
    _, _, numerical_gradient, _ = _math_ops()
    parameter = torch.tensor([0.4, -0.7], requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        return (value.square() * 0.5 + value.pow(3)).sum()

    expected = torch.autograd.grad(objective(parameter), parameter)[0]
    actual = numerical_gradient(objective, parameter.detach(), epsilon=1e-4)

    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
