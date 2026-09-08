#!/usr/bin/env python3
"""Generate text from a saved MiniGPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.corpus import load_character_corpus
from evaluation.generation import generate
from model.transformer import MiniGPT
from training.checkpoint import load_checkpoint
from training.config import load_config
from training.reproducibility import resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for checkpoint sampling."""
    parser = argparse.ArgumentParser(
        description="Generate text from a saved MiniGPT checkpoint.",
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to the final checkpoint in training.output_dir.",
    )
    parser.add_argument(
        "--prompt",
        default="The ",
        help="Text prompt used to start generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional override for evaluation.max_new_tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional override for evaluation.temperature; use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional override for evaluation.top_k.",
    )
    return parser


def _build_model(model_config: dict[str, object], device: torch.device) -> MiniGPT:
    """Construct a MiniGPT instance from the model configuration."""
    return MiniGPT(
        vocab_size=int(model_config["vocab_size"]),
        context_length=int(model_config["context_length"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        n_layers=int(model_config["n_layers"]),
        d_ff=int(model_config["d_ff"]),
    ).to(device)


def main(argv: list[str] | None = None) -> int:
    """Load a checkpoint and print one generated continuation."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]

    if data_config.get("tokenizer", "character") != "character":
        raise ValueError("this sampling command currently supports the character tokenizer only")
    if float(model_config.get("dropout", 0.0)) != 0.0:
        raise ValueError("the current MiniGPT implementation requires dropout=0.0")
    if model_config.get("position_encoding", "learned") != "learned":
        raise ValueError("the current MiniGPT implementation requires learned positions")

    tokenizer, _, _ = load_character_corpus(
        data_config["train_path"],
        data_config["validation_path"],
    )
    context_length = int(data_config["context_length"])
    model_context_length = int(model_config["context_length"])
    if context_length != model_context_length:
        raise ValueError("data and model context_length must match")

    model_vocab_size = int(model_config["vocab_size"])
    if tokenizer.vocab_size > model_vocab_size:
        raise ValueError("model vocab_size is smaller than the training vocabulary")

    model = _build_model(model_config, device)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        Path(training_config["output_dir"])
        / f"step-{int(training_config['max_steps']):06d}.pt"
    )
    checkpoint_step, _ = load_checkpoint(
        checkpoint_path,
        model,
        map_location=device,
    )

    if not args.prompt:
        raise ValueError("prompt must not be empty")

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(evaluation_config.get("max_new_tokens", 100))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(evaluation_config.get("temperature", 1.0))
    )
    top_k = (
        args.top_k
        if args.top_k is not None
        else evaluation_config.get("top_k")
    )
    if top_k is not None:
        top_k = int(top_k)

    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        allowed_vocab_size=tokenizer.vocab_size,
    )
    generated_text = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
    )

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint step: {checkpoint_step}")
    print(f"prompt: {args.prompt!r}")
    print("generated text:")
    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
