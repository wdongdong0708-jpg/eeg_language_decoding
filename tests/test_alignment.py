import pytest

from data.alignment import (
    AlignmentMethod,
    TimelineClaim,
    fixed_character_presentation_spans,
    validate_alignment_claim,
)


def test_fixed_pace_constructs_character_not_word_spans() -> None:
    spans = fixed_character_presentation_spans(
        "你，好！",
        onset_sec=1.0,
        pace_sec=0.35,
    )
    assert [span.character for span in spans] == ["你", "好"]
    assert [span.start_sec for span in spans] == pytest.approx([1.0, 1.35])
    assert [span.stop_sec for span in spans] == pytest.approx([1.35, 1.70])


def test_fixed_visual_pace_cannot_claim_speech_timing() -> None:
    with pytest.raises(ValueError, match="spoken audio"):
        validate_alignment_claim(
            method=AlignmentMethod.FIXED_CHARACTER_PRESENTATION,
            timeline=TimelineClaim.AUDIO_SPEECH,
        )
