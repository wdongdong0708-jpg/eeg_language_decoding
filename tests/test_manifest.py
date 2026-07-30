import pytest

from data.manifest import ManifestRecord, validate_manifest_records


def _record(*, record_id: str, subject_id: str, split: str) -> ManifestRecord:
    return ManifestRecord(
        record_id=record_id,
        dataset_id="chineseeeg2_pl",
        paradigm="passive_listening",
        book_id="littleprince",
        subject_id=subject_id,
        session_id="littleprince",
        run_id="11",
        block_id=f"{subject_id}-block-1",
        content_id="shared-content",
        eeg_path="example.vhdr",
        eeg_start_sample=0,
        eeg_stop_sample=250,
        sampling_rate_hz=250,
        text="测试。",
        stimulus_position=0,
        split=split,
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

