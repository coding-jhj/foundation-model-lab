#!/usr/bin/env python3
"""Run a small, reproducible MiniGPT pretraining experiment."""

from __future__ import annotations

import argparse

import torch

from data.corpus import load_character_corpus
from model.transformer import MiniGPT
from training.config import load_config
from training.pipeline import run_training
from training.reproducibility import resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for a MiniGPT training run."""
    parser = argparse.ArgumentParser(
        description="Train MiniGPT from a YAML experiment configuration.",
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML configuration file (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional short-run override for training.max_steps.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load data and configuration, then run one training experiment."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]

    if data_config.get("tokenizer", "character") != "character":
        raise ValueError("this training command currently supports the character tokenizer only")
    if float(model_config.get("dropout", 0.0)) != 0.0:
        raise ValueError("the current MiniGPT implementation requires dropout=0.0")
    if model_config.get("position_encoding", "learned") != "learned":
        raise ValueError("the current MiniGPT implementation requires learned positions")
    if int(training_config.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("gradient accumulation is not implemented yet")
    if bool(training_config.get("mixed_precision", False)):
        raise ValueError("mixed precision is not implemented yet")

    tokenizer, train_tokens, validation_tokens = load_character_corpus(
        data_config["train_path"],
        data_config["validation_path"],
    )
    context_length = int(data_config["context_length"])
    model_context_length = int(model_config["context_length"])
    if context_length != model_context_length:
        raise ValueError("data and model context_length must match")

    vocab_size = int(model_config["vocab_size"])
    if tokenizer.vocab_size > vocab_size:
        raise ValueError("model vocab_size is smaller than the training vocabulary")

    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=model_context_length,
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        n_layers=int(model_config["n_layers"]),
        d_ff=int(model_config["d_ff"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else int(training_config["max_steps"])
    )
    result = run_training(
        model,
        optimizer,
        train_tokens,
        validation_tokens,
        batch_size=int(training_config["batch_size"]),
        context_length=context_length,
        max_steps=max_steps,
        warmup_steps=int(training_config.get("warmup_steps", 0)),
        eval_interval=int(training_config.get("eval_interval", 0)),
        eval_batches=int(evaluation_config.get("eval_batches", 1)),
        checkpoint_interval=int(training_config.get("checkpoint_interval", 0)),
        output_dir=training_config.get("output_dir"),
        max_grad_norm=float(training_config["grad_clip_norm"])
        if training_config.get("grad_clip_norm") is not None
        else None,
    )

    print(f"device: {device}")
    print(f"vocabulary size: {tokenizer.vocab_size}")
    print(f"training tokens: {train_tokens.numel()}")
    print(f"validation tokens: {validation_tokens.numel()}")
    print(f"completed steps: {result.last_step}")
    print(f"final training loss: {result.train_losses[-1]:.6f}")
    if result.validation_losses:
        step, validation_loss = result.validation_losses[-1]
        print(f"validation loss at step {step}: {validation_loss:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
