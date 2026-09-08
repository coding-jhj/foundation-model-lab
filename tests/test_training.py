import pytest
import torch


def _batching_module():
    try:
        from data.batching import sample_language_model_batch
    except ModuleNotFoundError as error:
        pytest.fail(f"Expected data.batching to exist: {error}", pytrace=False)

    return sample_language_model_batch


def _training_module():
    try:
        from training.loop import language_model_loss, train, train_step
    except ModuleNotFoundError as error:
        pytest.fail(f"Expected training.loop to exist: {error}", pytrace=False)

    return language_model_loss, train, train_step


def _checkpoint_module():
    try:
        from training.checkpoint import load_checkpoint, save_checkpoint
    except ModuleNotFoundError as error:
        pytest.fail(f"Expected training.checkpoint to exist: {error}", pytrace=False)

    return load_checkpoint, save_checkpoint


def _tiny_model():
    from model.transformer import MiniGPT

    torch.manual_seed(0)
    return MiniGPT(
        vocab_size=16,
        context_length=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
    )


def test_sample_language_model_batch_aligns_inputs_and_next_token_targets() -> None:
    sample_language_model_batch = _batching_module()
    tokens = torch.arange(12)

    inputs, targets = sample_language_model_batch(
        tokens,
        batch_size=3,
        context_length=4,
    )

    assert inputs.shape == (3, 4)
    assert targets.shape == (3, 4)
    assert torch.equal(targets, inputs + 1)


def test_language_model_loss_returns_a_scalar() -> None:
    language_model_loss, _, _ = _training_module()
    model = _tiny_model()
    input_ids = torch.randint(0, 16, (2, 4))
    targets = torch.randint(0, 16, (2, 4))

    loss = language_model_loss(model, input_ids, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_train_step_updates_parameters_and_returns_a_finite_loss() -> None:
    _, _, train_step = _training_module()
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    input_ids = torch.randint(0, 16, (2, 4))
    targets = torch.randint(0, 16, (2, 4))
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_step(model, optimizer, input_ids, targets)

    parameters_changed = any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )
    assert isinstance(loss, float)
    assert loss > 0.0
    assert parameters_changed


def test_train_runs_the_requested_number_of_steps() -> None:
    _, train, _ = _training_module()
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    input_ids = torch.randint(0, 16, (2, 4))
    targets = torch.randint(0, 16, (2, 4))

    losses = train(
        model,
        optimizer,
        batch_provider=lambda: (input_ids, targets),
        num_steps=3,
    )

    assert len(losses) == 3
    assert all(loss > 0.0 for loss in losses)


def test_checkpoint_round_trip_restores_model_optimizer_and_metadata(tmp_path) -> None:
    load_checkpoint, save_checkpoint = _checkpoint_module()
    _, _, train_step = _training_module()
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    input_ids = torch.randint(0, 16, (2, 4))
    targets = torch.randint(0, 16, (2, 4))
    train_step(model, optimizer, input_ids, targets)

    checkpoint_path = tmp_path / "checkpoints" / "step-3.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=3,
        metadata={"experiment": "mini-gpt-smoke"},
    )

    restored_model = _tiny_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    step, metadata = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
    )

    assert step == 3
    assert metadata == {"experiment": "mini-gpt-smoke"}
    for parameter, restored_parameter in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(parameter, restored_parameter)
