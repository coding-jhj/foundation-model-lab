import pytest


def _schedule_module():
    try:
        from training.schedule import linear_warmup_scale
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected training.schedule to exist: {error}") from error

    return linear_warmup_scale


def test_linear_warmup_reaches_full_learning_rate_at_the_boundary() -> None:
    linear_warmup_scale = _schedule_module()

    assert linear_warmup_scale(step=0, warmup_steps=4) == 0.0
    assert linear_warmup_scale(step=2, warmup_steps=4) == 0.5
    assert linear_warmup_scale(step=4, warmup_steps=4) == 1.0
    assert linear_warmup_scale(step=9, warmup_steps=4) == 1.0


def test_linear_warmup_rejects_invalid_steps() -> None:
    linear_warmup_scale = _schedule_module()

    with pytest.raises(ValueError):
        linear_warmup_scale(step=-1, warmup_steps=4)
    with pytest.raises(ValueError):
        linear_warmup_scale(step=1, warmup_steps=-1)
