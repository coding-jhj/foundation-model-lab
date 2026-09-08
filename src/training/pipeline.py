from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from data.batching import sample_language_model_batch
from training.checkpoint import save_checkpoint
from training.loop import language_model_loss, train_step
from training.metrics import append_metrics_record, initialize_metrics_file
from training.schedule import linear_warmup_scale


@dataclass(frozen=True)
class TrainingRunResult:
    """Metrics collected during one fixed-length training run."""

    train_losses: list[float]
    validation_losses: list[tuple[int, float]]
    last_step: int
    elapsed_seconds: float = 0.0
    peak_gpu_memory_bytes: int = 0


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model must contain at least one parameter") from error


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    num_batches: int,
) -> float:
    """Estimate mean validation loss over randomly sampled token windows."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    was_training = model.training
    model.eval()
    device = _model_device(model)

    try:
        losses = []
        for _ in range(num_batches):
            input_ids, targets = sample_language_model_batch(
                tokens,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            losses.append(float(language_model_loss(model, input_ids, targets).cpu()))

        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    max_steps: int,
    warmup_steps: int = 0,
    eval_interval: int = 0,
    eval_batches: int = 1,
    checkpoint_interval: int = 0,
    output_dir: str | Path | None = None,
    metrics_path: str | Path | None = None,
    max_grad_norm: float | None = None,
) -> TrainingRunResult:
    """Run training, periodic validation, checkpointing, and metrics logging."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if eval_interval < 0 or checkpoint_interval < 0:
        raise ValueError("intervals must be non-negative")
    if eval_interval > 0 and eval_batches <= 0:
        raise ValueError("eval_batches must be positive when evaluation is enabled")

    train_losses = []
    validation_losses = []
    device = _model_device(model)
    base_learning_rates = [group["lr"] for group in optimizer.param_groups]
    metrics_file = initialize_metrics_file(metrics_path) if metrics_path is not None else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    for step in range(1, max_steps + 1):
        learning_rate_scale = linear_warmup_scale(step, warmup_steps)
        for group, base_learning_rate in zip(optimizer.param_groups, base_learning_rates):
            group["lr"] = base_learning_rate * learning_rate_scale

        input_ids, targets = sample_language_model_batch(
            train_tokens,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        train_loss = train_step(
            model,
            optimizer,
            input_ids,
            targets,
            max_grad_norm=max_grad_norm,
        )
        train_losses.append(train_loss)

        validation_loss = None
        if eval_interval > 0 and (step % eval_interval == 0 or step == max_steps):
            validation_loss = evaluate_loss(
                model,
                validation_tokens,
                batch_size=batch_size,
                context_length=context_length,
                num_batches=eval_batches,
            )
            validation_losses.append((step, validation_loss))

        if metrics_file is not None:
            metrics_record = {"step": step, "train_loss": train_loss}
            if validation_loss is not None:
                metrics_record["validation_loss"] = validation_loss
            append_metrics_record(metrics_file, metrics_record)

        if checkpoint_interval > 0 and (step % checkpoint_interval == 0 or step == max_steps):
            if output_dir is not None:
                checkpoint_metadata = {"train_loss": train_loss}
                if validation_loss is not None:
                    checkpoint_metadata["validation_loss"] = validation_loss
                checkpoint_path = Path(output_dir) / f"step-{step:06d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step=step,
                    metadata=checkpoint_metadata,
                )

    elapsed_seconds = time.perf_counter() - start_time
    peak_gpu_memory_bytes = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )

    if metrics_file is not None:
        append_metrics_record(
            metrics_file,
            {
                "event": "summary",
                "elapsed_seconds": elapsed_seconds,
                "last_step": max_steps,
                "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
                "steps_per_second": max_steps / max(elapsed_seconds, 1e-12),
            },
        )

    return TrainingRunResult(
        train_losses=train_losses,
        validation_losses=validation_losses,
        last_step=max_steps,
        elapsed_seconds=elapsed_seconds,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
    )
