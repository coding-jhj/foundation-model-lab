import torch


def _corpus_module():
    try:
        from data.corpus import encode_text_file
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected data.corpus to exist: {error}") from error

    return encode_text_file


def test_encode_text_file_uses_the_existing_training_tokenizer(tmp_path) -> None:
    encode_text_file = _corpus_module()
    from data.tokenizer import CharacterTokenizer

    tokenizer = CharacterTokenizer.from_texts(["abc"])
    test_path = tmp_path / "test.txt"
    test_path.write_text("abc?", encoding="utf-8")

    token_ids = encode_text_file(test_path, tokenizer)

    assert torch.equal(
        token_ids,
        torch.tensor([tokenizer.token_to_id(char) for char in "abc?"]),
    )
    assert token_ids.dtype == torch.long
