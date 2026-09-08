import random

import numpy as np
import pytest
import torch


def _reproducibility_module():
    try:
        from training.reproducibility import resolve_device, set_seed
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected training.reproducibility to exist: {error}") from error

    return resolve_device, set_seed


def test_set_seed_repeats_python_numpy_and_torch_randomness() -> None:
    _, set_seed = _reproducibility_module()

    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(3))

    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_resolve_device_supports_explicit_cpu_and_rejects_unknown_values() -> None:
    resolve_device, _ = _reproducibility_module()

    assert resolve_device("cpu").type == "cpu"

    with pytest.raises(ValueError):
        resolve_device("tpu")
