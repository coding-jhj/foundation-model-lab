import pytest


def _tokenizer_class():
    try:
        from data.tokenizer import CharacterTokenizer
    except ModuleNotFoundError as error:
        pytest.fail(f"Expected data.tokenizer to exist: {error}", pytrace=False)

    return CharacterTokenizer


def test_character_vocabulary_is_deterministic_and_has_special_tokens() -> None:
    CharacterTokenizer = _tokenizer_class()

    tokenizer = CharacterTokenizer.from_texts(["cab", "aa"])

    assert tokenizer.vocabulary == ("<pad>", "<unk>", "<bos>", "<eos>", "a", "b", "c")
    assert tokenizer.vocab_size == 7
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3


def test_encode_and_decode_round_trip_known_characters() -> None:
    CharacterTokenizer = _tokenizer_class()
    tokenizer = CharacterTokenizer.from_texts(["hello"])

    token_ids = tokenizer.encode("hello")

    assert tokenizer.decode(token_ids) == "hello"


def test_unknown_characters_use_the_unknown_token() -> None:
    CharacterTokenizer = _tokenizer_class()
    tokenizer = CharacterTokenizer.from_texts(["abc"])

    token_ids = tokenizer.encode("ab?")

    assert token_ids[-1] == tokenizer.unk_token_id
    assert tokenizer.decode(token_ids) == "ab<unk>"


def test_special_tokens_can_be_added_and_skipped_during_decoding() -> None:
    CharacterTokenizer = _tokenizer_class()
    tokenizer = CharacterTokenizer.from_texts(["ab"])

    token_ids = tokenizer.encode("ab", add_bos=True, add_eos=True)

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(token_ids, skip_special_tokens=True) == "ab"
