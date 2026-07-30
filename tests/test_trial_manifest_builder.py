from pathlib import Path

from data.text_alignment import (
    AlignmentTextUnit,
    align_monotonic_text_sequences,
    ensure_override_file,
)
from data.trial_manifest import read_event_pairs


def test_event_pairing_never_fabricates_or_crosses_boundaries(tmp_path: Path) -> None:
    events = tmp_path / "events.tsv"
    events.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0\t0\tROWE\t3\t5\n"
        "0\t0\tROWS\t4\t10\n"
        "0\t0\tROWS\t4\t12\n"
        "0\t0\tROWE\t3\t20\n"
        "0\t0\tROWS\t4\t30\n",
        encoding="utf-8",
    )
    result = read_event_pairs(events)
    assert [(pair.start_sample, pair.end_sample) for pair in result.pairs] == [
        (12, 20)
    ]
    assert result.orphan_starts == 2
    assert result.orphan_ends == 1


def test_monotonic_alignment_does_not_zip_truncate() -> None:
    source = [
        AlignmentTextUnit("s1", "甲", "甲"),
        AlignmentTextUnit("s2", "插入", "插入"),
        AlignmentTextUnit("s3", "乙", "乙"),
    ]
    target = [
        AlignmentTextUnit("t1", "甲", "甲", "g1"),
        AlignmentTextUnit("t2", "乙", "乙", "g2"),
    ]
    result = align_monotonic_text_sequences(source, target)
    assert result["s1"].global_text_id == "g1"
    assert result["s2"].status == "unresolved"
    assert result["s3"].global_text_id == "g2"


def test_empty_override_file_has_stable_schema(tmp_path: Path) -> None:
    path = tmp_path / "overrides.csv"
    ensure_override_file(path)
    assert path.read_text(encoding="utf-8").strip().split(",") == [
        "source_sentence_id",
        "target_sentence_id",
        "target_global_text_id",
        "decision",
        "score",
        "reviewer",
        "note",
    ]
