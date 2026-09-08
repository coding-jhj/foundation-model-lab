import torch
from torch import nn


def test_count_parameters_can_count_trainable_and_all_parameters() -> None:
    from training.model_stats import count_parameters

    model = nn.Linear(3, 2)
    model.bias.requires_grad = False

    assert count_parameters(model) == 6
    assert count_parameters(model, trainable_only=False) == 8
