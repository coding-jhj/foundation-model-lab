import json

import torch


def _pipeline_module():
    try:
        from training.pipeline import run_training
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected training.pipeline to exist: {error}") from error

    return run_training


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


def test_run_training_writes_step_metrics_and_a_summary_record(tmp_path) -> None:
    run_training = _pipeline_module()
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    train_tokens = torch.arange(64) % 16
    validation_tokens = torch.arange(48) % 16
    metrics_path = tmp_path / "run" / "metrics.jsonl"

    result = run_training(
        model,
        optimizer,
        train_tokens,
        validation_tokens,
        batch_size=2,
        context_length=4,
        max_steps=3,
        eval_interval=2,
        eval_batches=1,
        metrics_path=metrics_path,
    )

    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    step_records = [record for record in records if "step" in record]
    summary_records = [record for record in records if record.get("event") == "summary"]

    assert [record["step"] for record in step_records] == [1, 2, 3]
    assert len(summary_records) == 1
    assert summary_records[0]["last_step"] == 3
    assert summary_records[0]["elapsed_seconds"] > 0.0
    assert result.elapsed_seconds > 0.0
