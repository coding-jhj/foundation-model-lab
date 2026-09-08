from scripts.prepare_wikitext2 import serialize_documents


def test_serialize_documents_discards_empty_rows_and_preserves_boundaries() -> None:
    documents = [" first article ", "", "second article", "   "]

    serialized = serialize_documents(documents)

    assert serialized == "first article\n\nsecond article\n"
