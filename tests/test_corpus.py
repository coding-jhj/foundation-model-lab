import torch


def _corpus_module():
    try:
        from data.corpus import load_character_corpus
    except ModuleNotFoundError as error:
        raise AssertionError(f"Expected data.corpus to exist: {error}") from error

    return load_character_corpus


def test_character_corpus_builds_vocabulary_from_training_text_only(tmp_path) -> None:
    load_character_corpus = _corpus_module()
    train_path = tmp_path / "train.txt"
    validation_path = tmp_path / "validation.txt"
    train_path.write_text("abcabc", encoding="utf-8")
    validation_path.write_text("abc?", encoding="utf-8")

    tokenizer, train_tokens, validation_tokens = load_character_corpus(
        train_path,
        validation_path,
    )

    assert tokenizer.vocab_size == 7
    assert train_tokens.dtype == torch.long
    assert validation_tokens[-1].item() == tokenizer.unk_token_id
