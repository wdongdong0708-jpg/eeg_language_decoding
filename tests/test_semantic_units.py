from data.semantic_units import semantic_unit_annotations


def _surfaces(text: str) -> set[str]:
    return {
        text[start:stop]
        for start, stop in semantic_unit_annotations(text)
    }


def test_semantic_units_include_words_and_two_content_token_phrases() -> None:
    text = "\u5c0f\u738b\u5b50\u770b\u65e5\u843d"
    surfaces = _surfaces(text)
    assert "\u5c0f\u738b\u5b50" in surfaces
    assert "\u770b\u65e5\u843d" in surfaces


def test_semantic_units_never_cross_punctuation() -> None:
    text = "\u73ab\u7470\uff0c\u770b\u65e5\u843d"
    surfaces = _surfaces(text)
    assert "\u73ab\u7470" in surfaces
    assert all("\uff0c" not in surface for surface in surfaces)
