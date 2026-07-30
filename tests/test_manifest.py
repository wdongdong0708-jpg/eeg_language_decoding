import pytest

from data.manifest import (
    EEG_END_SAMPLE_SEMANTICS,
    MANIFEST_SCHEMA_VERSION,
    ManifestRecord,
    validate_manifest_records,
)
from data.trial_manifest import DEFAULT_SPLIT_SEED


def _record(*, record_id: str, subject_id: str, split: str) -> ManifestRecord:
    return ManifestRecord(
        dataset_version="ChineseEEG2",
        paradigm="passive_listening",
        subject_id=subject_id,
        session_id="littleprince",
        book_id="littleprince",
        chapter_id=1,
        sentence_id="sentence-v1-example",
        global_text_id="global-text-v1-example",
        raw_text="测试。",
        normalized_text="测试。",
        text_hash="sha256:example",
        raw_text_hash="sha256:raw",
        normalized_text_hash="sha256:example",
        text_hash_basis="normalized_text/sha256-utf8-v1",
        char_count=3,
        raw_char_count=3,
        highlight_char_count=2,
        char_count_method="test",
        word_count=1,
        word_count_method="test",
        eeg_file="example.eeg",
        eeg_start_sample=0,
        eeg_end_sample=250,
        eeg_end_sample_semantics=EEG_END_SAMPLE_SEMANTICS,
        eeg_duration_sec=1.0,
        eeg_sampling_rate=250.0,
        audio_file=None,
        audio_start_sec=None,
        audio_end_sec=None,
        audio_duration_sec=None,
        event_source="events.tsv",
        alignment_source="test",
        quality_flag="ok",
        split_group_id="split-group-v1-shared-content",
        split=split,
        record_id=record_id,
        run_id="11",
        block_id=f"{subject_id}-block-1",
        content_id="shared-content",
        material_variant="f1",
        speaker_id="f1",
        stimulus_position=0,
        source_excel_file="stimulus.xlsx",
        source_excel_row=2,
        normalization_version="text-normalization-v2",
        normalization_trace="[]",
        matching_alias=None,
        matching_alias_method="none",
        text_alignment_status="exact",
        text_alignment_score=100.0,
        global_text_alignment_status="exact",
        global_text_alignment_score=100.0,
        audio_alignment_method="null_unverified",
        audio_alignment_evidence="test",
        event_pair_index=0,
        preceding_chapter_event="CH01",
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        split_seed=DEFAULT_SPLIT_SEED,
    )


def test_same_content_across_subjects_may_share_one_split() -> None:
    validate_manifest_records(
        [
            _record(record_id="r1", subject_id="sub-01", split="test"),
            _record(record_id="r2", subject_id="sub-02", split="test"),
        ]
    )


def test_same_content_across_subjects_cannot_cross_splits() -> None:
    with pytest.raises(ValueError, match="multiple splits"):
        validate_manifest_records(
            [
                _record(record_id="r1", subject_id="sub-01", split="train"),
                _record(record_id="r2", subject_id="sub-02", split="test"),
            ]
        )


def test_manifest_uses_half_open_interval() -> None:
    record = _record(record_id="r1", subject_id="sub-01", split="test")
    assert record.eeg_duration_sec == (
        record.eeg_end_sample - record.eeg_start_sample
    ) / record.eeg_sampling_rate

