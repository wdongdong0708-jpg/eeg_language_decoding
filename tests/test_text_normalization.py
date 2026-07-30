from data.text_normalization import (
    make_content_id,
    normalize_text,
    normalize_text_with_trace,
)


def test_normalization_is_unicode_and_whitespace_stable() -> None:
    assert normalize_text("Ａ B\u3000Ｃ") == "ABC"
    # Repeated punctuation is intentionally preserved, not aggressively merged.
    assert normalize_text("“你好” ……") == '"你好"……'


def test_raw_text_and_rule_trace_are_preserved() -> None:
    result = normalize_text_with_trace("Ａ \u200bB\r\nＣ")
    assert result.raw_text == "Ａ \u200bB\r\nＣ"
    assert result.normalized_text == "ABC"
    rules = {change.rule for change in result.changes}
    assert {
        "newline_style_to_lf",
        "remove_invisible_format_character",
        "fullwidth_ascii_to_ascii",
        "remove_linebreak",
        "remove_whitespace",
    } <= rules


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
