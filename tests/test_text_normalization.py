from data.text_normalization import make_content_id, normalize_text


def test_normalization_is_unicode_and_whitespace_stable() -> None:
    assert normalize_text("Ａ B\u3000Ｃ") == "ABC"
    assert normalize_text("“你好” ……") == '"你好"…'


def test_content_id_is_independent_of_dataset_occurrence() -> None:
    first = make_content_id(book_id="LittlePrince", text="  你好，世界。 ")
    second = make_content_id(book_id="littleprince", text="你好，世界。")
    assert first == second


def test_reviewed_sequence_key_can_distinguish_repeated_occurrences() -> None:
    first = make_content_id(
        book_id="littleprince",
        text="他说。",
        sequence_key="chapter-01/sentence-001",
    )
    second = make_content_id(
        book_id="littleprince",
        text="他说。",
        sequence_key="chapter-02/sentence-001",
    )
    assert first != second
