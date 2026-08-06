"""Frozen text-only semantic-unit annotations for character spans."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from data.alignment import is_highlighted_character

SEMANTIC_UNIT_RULE_VERSION = (
    "jieba-0.42.1-precise-hmm_false-content-pos-single-or-two-token-v1"
)
CONTENT_POS_INITIALS = frozenset("nvadtsfilz")


@dataclass(frozen=True, slots=True)
class SemanticUnitAnnotation:
    raw_start: int
    raw_stop: int
    kind: str
    pos_pattern: str
    token_count: int


@dataclass(frozen=True, slots=True)
class _Token:
    word: str
    flag: str
    raw_start: int
    raw_stop: int

    @property
    def is_content(self) -> bool:
        return bool(self.flag) and self.flag[0] in CONTENT_POS_INITIALS


def _contains_only_clock_characters(text: str) -> bool:
    return bool(text) and all(is_highlighted_character(character) for character in text)


@lru_cache(maxsize=50_000)
def semantic_unit_annotations(
    text: str,
    *,
    minimum_characters: int = 2,
    maximum_characters: int = 5,
) -> dict[tuple[int, int], SemanticUnitAnnotation]:
    """Mark content words and adjacent two-content-token phrases.

    The rule is text-only and frozen before model training. Punctuation,
    whitespace, function words on their own, and phrases crossing punctuation
    are excluded.
    """

    if minimum_characters < 1 or maximum_characters < minimum_characters:
        raise ValueError("Invalid semantic-unit character bounds")
    import jieba.posseg as pseg

    tokens: list[_Token] = []
    cursor = 0
    for pair in pseg.cut(text, HMM=False):
        word = str(pair.word)
        if not word:
            continue
        start = cursor
        stop = start + len(word)
        if text[start:stop] != word:
            start = text.find(word, cursor)
            if start < 0:
                raise ValueError("jieba token offsets cannot be reconciled with text")
            stop = start + len(word)
        tokens.append(_Token(word, str(pair.flag), start, stop))
        cursor = stop

    output: dict[tuple[int, int], SemanticUnitAnnotation] = {}
    for token in tokens:
        length = len(token.word)
        if (
            token.is_content
            and minimum_characters <= length <= maximum_characters
            and _contains_only_clock_characters(token.word)
        ):
            output[(token.raw_start, token.raw_stop)] = SemanticUnitAnnotation(
                raw_start=token.raw_start,
                raw_stop=token.raw_stop,
                kind="jieba_content_word",
                pos_pattern=token.flag,
                token_count=1,
            )
    for left, right in zip(tokens, tokens[1:], strict=False):
        if not left.is_content or not right.is_content:
            continue
        if left.raw_stop != right.raw_start:
            continue
        phrase = text[left.raw_start : right.raw_stop]
        length = len(phrase)
        if not (
            minimum_characters <= length <= maximum_characters
            and _contains_only_clock_characters(phrase)
        ):
            continue
        output.setdefault(
            (left.raw_start, right.raw_stop),
            SemanticUnitAnnotation(
                raw_start=left.raw_start,
                raw_stop=right.raw_stop,
                kind="jieba_two_content_token_phrase",
                pos_pattern=f"{left.flag}+{right.flag}",
                token_count=2,
            ),
        )
    return output
