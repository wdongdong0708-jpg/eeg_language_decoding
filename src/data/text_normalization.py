"""Canonical text normalization and subject-independent content identities."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

CONTENT_ID_VERSION = "content-v1"

_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\.{3,}")
_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‛": "'",
        "﹐": "，",
        "﹑": "、",
        "﹒": "。",
        "⋯": "…",
    }
)


def normalize_text(text: str) -> str:
    """Normalize representation without inventing or removing linguistic units."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_PUNCTUATION_TRANSLATION)
    normalized = normalized.replace("……", "…")
    normalized = _ELLIPSIS_RE.sub("…", normalized)
    normalized = _WHITESPACE_RE.sub("", normalized)
    return normalized.strip()


def normalize_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"identifier must be str, got {type(value).__name__}")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = _WHITESPACE_RE.sub("-", normalized)
    if not normalized:
        raise ValueError("identifier cannot be empty")
    return normalized


def make_content_id(
    *,
    book_id: str,
    text: str,
    sequence_key: str | None = None,
    canonical_audio_id: str | None = None,
) -> str:
    """Create a stable content ID without subject, session or paradigm fields.

    ``sequence_key`` may distinguish repeated occurrences only when it is itself
    canonical across datasets and paradigms (for example, a reviewed
    book/chapter/sentence index). Omitting it conservatively groups identical
    normalized text together.
    """

    normalized_text = normalize_text(text)
    if not normalized_text and canonical_audio_id is None:
        raise ValueError("content identity requires non-empty text or canonical_audio_id")

    payload = {
        "version": CONTENT_ID_VERSION,
        "book_id": normalize_identifier(book_id),
        "text": normalized_text,
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
