from __future__ import annotations

from pathlib import Path

import torch

from data.tokenizer import CharacterTokenizer


def encode_text_file(
    path: str | Path,
    tokenizer: CharacterTokenizer,
) -> torch.Tensor:
    """Read a UTF-8 text file and encode it with an existing tokenizer."""
    text = Path(path).read_text(encoding="utf-8")
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def load_character_corpus(
    train_path: str | Path,
    validation_path: str | Path,
) -> tuple[CharacterTokenizer, torch.Tensor, torch.Tensor]:
    """Load text files and encode validation text with a training-only vocabulary."""
    train_text = Path(train_path).read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_texts([train_text])
    train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation_tokens = encode_text_file(validation_path, tokenizer)
    return tokenizer, train_tokens, validation_tokens
