import torch


def _pipeline_module():
    try:
        from training.pipeline import (
            TrainingRunResult,
            evaluate_loss,
            run_training,
        )
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected training.pipeline to exist: {error}") from error

    return TrainingRunResult, evaluate_loss, run_training


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


def test_evaluate_loss_returns_a_finite_scalar() -> None:
    _, evaluate_loss, _ = _pipeline_module()
    model = _tiny_model()
    tokens = torch.arange(32) % 16

    loss = evaluate_loss(
        model,
        tokens,
        batch_size=2,
        context_length=4,
        num_batches=2,
    )

    assert isinstance(loss, float)
    assert loss > 0.0


def test_run_training_records_metrics_and_writes_checkpoints(tmp_path) -> None:
    TrainingRunResult, _, run_training = _pipeline_module()
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train_tokens = torch.arange(64) % 16
    validation_tokens = torch.arange(48) % 16

    result = run_training(
        model,
        optimizer,
        train_tokens,
        validation_tokens,
        batch_size=2,
        context_length=4,
        max_steps=3,
        eval_interval=2,
        checkpoint_interval=3,
        output_dir=tmp_path,
    )

    assert isinstance(result, TrainingRunResult)
    assert result.last_step == 3
    assert len(result.train_losses) == 3
    assert [step for step, _ in result.validation_losses] == [2, 3]
    assert (tmp_path / "step-000003.pt").exists()
