from pathlib import Path

import pandas as pd

from data.chineseeeg1_timeline import (
    OFFICIAL_SKIPPED_CHARACTERS,
    audit_chineseeeg1_timeline,
    chineseeeg1_clock_positions,
    inspect_official_presentation_code,
)


def _official_source(path: Path) -> None:
    punctuation = repr(list(OFFICIAL_SKIPPED_CHARACTERS))
    path.write_text(
        f"""
def calculate_length_without_punctuation_and_indexes(sentence):
    punctuations = {punctuation}
    return sum(char not in punctuations for char in sentence)

def run(routineTimer, eci_client, args):
    eci_client.send_event(event_type='ROWS')
    while routineTimer.getTime() < args.shift_time:
        pass
    eci_client.send_event(event_type='ROWE')

parser.add_argument('--shift_time', type=float, default=0.35)
""",
        encoding="utf-8",
    )


def test_official_character_rule_is_not_generic_unicode_punctuation() -> None:
    text = "甲，乙—丙\n丁"
    assert chineseeeg1_clock_positions(text) == (0, 2, 3, 4, 6)


def test_audit_verifies_approximate_but_not_exact_onsets(tmp_path: Path) -> None:
    source = tmp_path / "play_novel.py"
    _official_source(source)
    evidence = inspect_official_presentation_code(source)
    rows = []
    for run in ("01", "02"):
        for count in range(2, 11):
            rows.append(
                {
                    "dataset_version": "ChineseEEG1",
                    "paradigm": "silent_reading",
                    "raw_text": "甲" * count,
                    "highlight_char_count": count,
                    "eeg_duration_sec": -0.05 + 0.425 * count,
                    "subject_id": "01",
                    "session_id": "book",
                    "run_id": run,
                }
            )
    manifest = tmp_path / "manifest.bin"
    manifest.write_bytes(b"manifest")
    report = audit_chineseeeg1_timeline(
        pd.DataFrame(rows),
        code_evidence=evidence,
        manifest_path=manifest,
    )
    assert report["verdict"] == "verified_approximate_only"
    assert report["configured_pace_verified"] is True
    assert report["exact_character_onsets_verified"] is False
