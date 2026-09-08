#!/usr/bin/env python3
"""Run the controlled three-variant MiniGPT scaling pilot."""

from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_CONFIGS = (
    "configs/wikitext2_scaling_small.yaml",
    "configs/wikitext2_scaling_base.yaml",
    "configs/wikitext2_scaling_large.yaml",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the scaling pilot."""
    parser = argparse.ArgumentParser(
        description="Run the three controlled WikiText-2 model-size experiments.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Configuration files to run in order.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional step override applied to every configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run each configuration in an isolated Python process."""
    args = build_parser().parse_args(argv)

    for config_path in args.configs:
        print(f"=== scaling run: {config_path} ===", flush=True)
        command = [
            sys.executable,
            "scripts/train_mini_gpt.py",
            "--config",
            config_path,
        ]
        if args.max_steps is not None:
            command.extend(["--max-steps", str(args.max_steps)])

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
