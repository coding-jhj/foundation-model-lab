from __future__ import annotations

from collections.abc import Iterable, Sequence


DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


class CharacterTokenizer:
    """Map characters to deterministic token IDs and back."""

    def __init__(
        self,
        vocabulary: Sequence[str],
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self._vocabulary = tuple(vocabulary)
        self._special_tokens = tuple(special_tokens)

        if len(set(self._vocabulary)) != len(self._vocabulary):
            raise ValueError("vocabulary entries must be unique")
        if len(set(self._special_tokens)) != len(self._special_tokens):
            raise ValueError("special tokens must be unique")
        if not set(self._special_tokens).issubset(self._vocabulary):
            raise ValueError("special tokens must be included in the vocabulary")

        self._token_to_id = {
            token: token_id for token_id, token in enumerate(self._vocabulary)
        }

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> CharacterTokenizer:
        """Build a tokenizer from the sorted set of observed characters."""
        special_token_set = set(special_tokens)
        characters = {character for text in texts for character in text}
        vocabulary = tuple(special_tokens) + tuple(sorted(characters - special_token_set))
        return cls(vocabulary=vocabulary, special_tokens=special_tokens)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._vocabulary

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id("<pad>")

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id("<unk>")

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id("<eos>")

    def token_to_id(self, token: str) -> int:
        """Return the ID for a token or the unknown-token ID when absent."""
        return self._token_to_id.get(token, self.unk_token_id)

    def id_to_token(self, token_id: int) -> str:
        """Return the token for an ID and reject IDs outside the vocabulary."""
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token ID {token_id} is outside the vocabulary")
        return self._vocabulary[token_id]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text into character IDs with optional boundary tokens."""
        token_ids = [self.token_to_id(character) for character in text]

        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)

        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs into text, optionally omitting special tokens."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if skip_special_tokens and token in self._special_tokens:
                continue
            tokens.append(token)

        return "".join(tokens)

    def __len__(self) -> int:
        return self.vocab_size
