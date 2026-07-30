"""Alignment evidence types with explicit protection against false timestamps."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum


class AlignmentMethod(str, Enum):
    EVENT_ROW_BOUNDARY = "event_row_boundary"
    FIXED_CHARACTER_PRESENTATION = "fixed_character_presentation"
    AUDIO_FORCED_ALIGNMENT = "audio_forced_alignment"
    ASR_TIMESTAMP = "asr_timestamp"
    MONOTONIC_WEAK_ALIGNMENT = "monotonic_weak_alignment"


class TimelineClaim(str, Enum):
    VISUAL_PRESENTATION = "visual_presentation"
    AUDIO_SPEECH = "audio_speech"
    EEG_RECORDING = "eeg_recording"


@dataclass(frozen=True, slots=True)
class CharacterPresentationSpan:
    character: str
    source_index: int
    start_sec: float
    stop_sec: float


_NON_HIGHLIGHTED = frozenset(
    "\n\r\t 。，“”！？：；、》《.（）…·,'\"!?;:()[]{}—-"
)


def is_highlighted_character(character: str) -> bool:
    """Return whether the experiment presentation advances on this character."""

    if len(character) != 1:
        raise ValueError("Expected exactly one Unicode character")
    category = unicodedata.category(character)
    return character not in _NON_HIGHLIGHTED and not category.startswith(
        ("P", "Z", "C")
    )


def validate_alignment_claim(
    *,
    method: AlignmentMethod,
    timeline: TimelineClaim,
) -> None:
    """Reject presentation timing presented as measured speech timing."""

    if (
        method is AlignmentMethod.FIXED_CHARACTER_PRESENTATION
        and timeline is TimelineClaim.AUDIO_SPEECH
    ):
        raise ValueError(
            "Fixed visual character pace cannot be claimed as spoken audio timing"
        )


def fixed_character_presentation_spans(
    text: str,
    *,
    onset_sec: float,
    pace_sec: float,
) -> list[CharacterPresentationSpan]:
    """Construct visual highlight spans only; this function does not segment words."""

    if onset_sec < 0 or pace_sec <= 0:
        raise ValueError("onset_sec must be non-negative and pace_sec must be positive")
    validate_alignment_claim(
        method=AlignmentMethod.FIXED_CHARACTER_PRESENTATION,
        timeline=TimelineClaim.VISUAL_PRESENTATION,
    )

    spans: list[CharacterPresentationSpan] = []
    step = 0
    for source_index, character in enumerate(text):
        if not is_highlighted_character(character):
            continue
        start = onset_sec + step * pace_sec
        spans.append(
            CharacterPresentationSpan(
                character=character,
                source_index=source_index,
                start_sec=start,
                stop_sec=start + pace_sec,
            )
        )
        step += 1
    return spans
