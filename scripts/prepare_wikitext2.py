#!/usr/bin/env python3
"""Download and serialize the WikiText-2 raw language-modeling dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLITS = ("train", "validation", "test")
DATASET_CARD_URL = "https://huggingface.co/datasets/Salesforce/wikitext"


def serialize_documents(documents: Iterable[str]) -> str:
    """Normalize non-empty documents and separate them with blank lines."""
    normalized_documents = [document.strip() for document in documents if document.strip()]
    if not normalized_documents:
        return ""

    return "\n\n".join(normalized_documents) + "\n"


def prepare_wikitext2(
    output_dir: str | Path,
    revision: str = "main",
) -> dict[str, Any]:
    """Download WikiText-2 raw splits and write text files plus metadata."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install the data extra first: python -m pip install -e ".[data]""
        ) from error

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=revision)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": revision,
        "dataset_card": DATASET_CARD_URL,
        "license": ["cc-by-sa-3.0", "gfdl"],
        "splits": {},
    }

    for split_name in SPLITS:
        split_dataset = dataset[split_name]
        serialized = serialize_documents(row["text"] for row in split_dataset)
        split_path = destination / f"{split_name}.txt"
        split_path.write_text(serialized, encoding="utf-8")
        metadata["splits"][split_name] = {
            "examples": len(split_dataset),
            "characters": len(serialized),
            "path": str(split_path),
        }

    metadata_path = destination / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare the raw WikiText-2 dataset for character-level training.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/wikitext-2-raw-v1",
        help="Directory for serialized splits and metadata.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision to load.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare the dataset and print a compact summary."""
    args = build_parser().parse_args(argv)
    metadata = prepare_wikitext2(args.output_dir, revision=args.revision)

    print(f"dataset: {metadata['dataset']}")
    print(f"config: {metadata['config']}")
    print(f"revision: {metadata['revision']}")
    for split_name, split_metadata in metadata["splits"].items():
        print(
            f"{split_name}: {split_metadata['examples']} examples, "
            f"{split_metadata['characters']} characters",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
