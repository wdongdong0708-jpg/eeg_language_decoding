"""Versioned, auditable text normalization and content identities.

The exact source cell text is never mutated.  Normalization returns a separate
value and a machine-readable trace of every rule that changed it.  No
simplified/traditional Chinese conversion is performed.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

NORMALIZATION_VERSION = "text-normalization-v2"
CONTENT_ID_VERSION = "content-v2"
TEXT_HASH_ALGORITHM = "sha256-utf8-v1"
CHAR_COUNT_METHOD = "unicode-codepoints-excluding-whitespace-v1"
RAW_CHAR_COUNT_METHOD = "unicode-codepoints-v1"
HIGHLIGHT_CHAR_COUNT_METHOD = "unicode-nonpunctuation-nonspace-codepoints-v1"

_NEWLINE_RE = re.compile(r"\r\n?|\n")
_INVISIBLE_FORMAT_CHARACTERS = frozenset(
    {
        "\u00ad",  # soft hyphen
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\ufeff",  # BOM / zero-width no-break space
    }
)
_CHARACTER_TRANSLATION = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‛": "'",
    "′": "'",
    "〈": "《",
    "〉": "》",
    "﹤": "《",
    "﹥": "》",
    "﹐": "，",
    "﹑": "、",
    "﹒": "。",
    "⋯": "…",
}


@dataclass(frozen=True, slots=True)
class NormalizationChange:
    rule: str
    count: int

    def to_dict(self) -> dict[str, object]:
        return {"rule": self.rule, "count": self.count}


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    raw_text: str
    normalized_text: str
    version: str
    changes: tuple[NormalizationChange, ...]
    raw_text_hash: str
    normalized_text_hash: str

    @property
    def trace_json(self) -> str:
        return json.dumps(
            [change.to_dict() for change in self.changes],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def stable_text_hash(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _replace_newlines(text: str) -> tuple[str, int]:
    matches = list(_NEWLINE_RE.finditer(text))
    return _NEWLINE_RE.sub("\n", text), len(matches)


def _remove_invisible_format_characters(text: str) -> tuple[str, int]:
    removed = sum(character in _INVISIBLE_FORMAT_CHARACTERS for character in text)
    return (
        "".join(
            character
            for character in text
            if character not in _INVISIBLE_FORMAT_CHARACTERS
        ),
        removed,
    )


def _normalize_fullwidth_ascii(text: str) -> tuple[str, int]:
    output: list[str] = []
    changed = 0
    for character in text:
        codepoint = ord(character)
        if 0xFF01 <= codepoint <= 0xFF5E:
            output.append(chr(codepoint - 0xFEE0))
            changed += 1
        elif character == "\u3000":
            output.append(" ")
            changed += 1
        else:
            output.append(character)
    return "".join(output), changed


def _translate_characters(text: str) -> tuple[str, int]:
    output: list[str] = []
    changed = 0
    for character in text:
        replacement = _CHARACTER_TRANSLATION.get(character, character)
        output.append(replacement)
        changed += replacement != character
    return "".join(output), changed


def _remove_linebreaks(text: str) -> tuple[str, int]:
    count = text.count("\n")
    return text.replace("\n", ""), count


def _remove_remaining_whitespace(text: str) -> tuple[str, int]:
    removed = sum(character.isspace() for character in text)
    return "".join(character for character in text if not character.isspace()), removed


def normalize_text_with_trace(text: str) -> NormalizationResult:
    """Return a conservative normalized alias and exact transformation trace."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    working = text
    changes: list[NormalizationChange] = []

    working, count = _replace_newlines(working)
    if count and working != text:
        changes.append(NormalizationChange("newline_style_to_lf", count))

    working, count = _remove_invisible_format_characters(working)
    if count:
        changes.append(NormalizationChange("remove_invisible_format_character", count))

    nfc = unicodedata.normalize("NFC", working)
    if nfc != working:
        changes.append(NormalizationChange("unicode_nfc", 1))
        working = nfc

    working, count = _normalize_fullwidth_ascii(working)
    if count:
        changes.append(NormalizationChange("fullwidth_ascii_to_ascii", count))

    working, count = _translate_characters(working)
    if count:
        changes.append(NormalizationChange("punctuation_variant_canonicalization", count))

    working, count = _remove_linebreaks(working)
    if count:
        changes.append(NormalizationChange("remove_linebreak", count))

    working, count = _remove_remaining_whitespace(working)
    if count:
        changes.append(NormalizationChange("remove_whitespace", count))

    return NormalizationResult(
        raw_text=text,
        normalized_text=working,
        version=NORMALIZATION_VERSION,
        changes=tuple(changes),
        raw_text_hash=stable_text_hash(text),
        normalized_text_hash=stable_text_hash(working),
    )


def normalize_text(text: str) -> str:
    """Compatibility wrapper returning only the versioned normalized value."""

    return normalize_text_with_trace(text).normalized_text


def normalize_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"identifier must be str, got {type(value).__name__}")
    normalized = unicodedata.normalize("NFC", value).strip().casefold()
    normalized = "-".join(normalized.split())
    if not normalized:
        raise ValueError("identifier cannot be empty")
    return normalized


def raw_char_count(text: str) -> int:
    return len(text)


def non_whitespace_char_count(text: str) -> int:
    return sum(not character.isspace() for character in text)


def highlight_char_count(text: str) -> int:
    """Count visual-clock characters, excluding punctuation/space/control."""

    return sum(
        not unicodedata.category(character).startswith(("P", "Z", "C"))
        for character in text
    )


@lru_cache(maxsize=1)
def jieba_word_count_method() -> str:
    import jieba

    dictionary_handle = jieba.dt.get_dict_file()
    dictionary_path = Path(dictionary_handle.name)
    dictionary_hash = hashlib.sha256(dictionary_path.read_bytes()).hexdigest()[:16]
    return (
        f"jieba-{jieba.__version__}-precise-hmm_false-"
        f"default_dict_sha256_{dictionary_hash}-exclude_punct_space-v1"
    )


@lru_cache(maxsize=65_536)
def deterministic_word_count(text: str) -> int:
    """Count jieba 0.42.1 precise-mode tokens; never BERT WordPieces."""

    import jieba

    jieba.setLogLevel(jieba.logging.ERROR)
    tokens = jieba.cut(text, cut_all=False, HMM=False)
    return sum(
        1
        for token in tokens
        if token
        and any(
            not unicodedata.category(character).startswith(("P", "Z", "C"))
            for character in token
        )
    )


def make_content_id(
    *,
    book_id: str,
    text: str,
    sequence_key: str | None = None,
    canonical_audio_id: str | None = None,
) -> str:
    """Create a stable content ID without subject/session/paradigm fields."""

    normalized_text = normalize_text(text)
    if not normalized_text and canonical_audio_id is None:
        raise ValueError("content identity requires non-empty text or canonical_audio_id")
    payload = {
        "version": CONTENT_ID_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "book_id": normalize_identifier(book_id),
        "normalized_text": normalized_text,
        "sequence_key": (
            normalize_identifier(sequence_key) if sequence_key is not None else None
        ),
        "canonical_audio_id": (
            normalize_identifier(canonical_audio_id)
            if canonical_audio_id is not None
            else None
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{CONTENT_ID_VERSION}-{digest[:24]}"
