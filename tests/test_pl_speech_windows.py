from __future__ import annotations

from pathlib import Path

from data.pl_speech import (
    PLSpeechWindowSpec,
    build_pl_speech_windows,
    load_pl_window_jsonl,
    write_pl_window_jsonl,
)


def _row(
    index: int,
    *,
    duration_sec: float = 6.0,
    audio_file: str | None = "audio.wav",
) -> dict[str, object]:
    return {
        "record_id": f"record-{index}",
        "block_id": f"block-{index}",
        "split_group_id": f"group-{index}",
        "dataset_version": "ChineseEEG2",
        "paradigm": "passive_listening",
        "subject_id": f"{index:02d}",
        "session_id": "littleprince",
        "run_id": "11",
        "speaker_id": "f1",
        "eeg_file": f"recording-{index}.eeg",
        "eeg_sampling_rate": 250.0,
        "eeg_start_sample": 1000,
        "eeg_end_sample": 1000 + round(duration_sec * 250),
        "audio_file": audio_file,
        "audio_start_sec": 1.0 if audio_file else None,
        "audio_end_sec": 1.0 + duration_sec if audio_file else None,
        "audio_alignment_method": "pl-audio-events-v1",
        "audio_alignment_evidence": "synthetic",
        "quality_flag": "ok",
        "stimulus_position": index,
        "char_count": 12,
        "eeg_duration_sec": duration_sec,
    }


def test_nonoverlap_windows_share_physical_span_and_inherit_split() -> None:
    rows = [_row(1)]
    windows, audit = build_pl_speech_windows(
        rows,
        record_partitions={"record-1": "test"},
        spec=PLSpeechWindowSpec(window_sec=3.0, stride_sec=3.0, delay_ms=0),
        manifest_path="manifest.parquet",
        split_artifact_path="splits.json",
        audio_info_provider=lambda _: (12_000, 120_000),
    )
    windows = sorted(windows, key=lambda item: item.eeg_start_sample)
    assert len(windows) == 2
    assert [(item.eeg_start_sample, item.eeg_stop_sample) for item in windows] == [
        (1000, 1750),
        (1750, 2500),
    ]
    assert [(item.audio_start_sec, item.audio_stop_sec) for item in windows] == [
        (1.0, 4.0),
        (4.0, 7.0),
    ]
    assert {item.split for item in windows} == {"test"}
    assert audit["leakage_checks"]["content_group_cross_partition_count"] == 0



def test_brainmagick_float_condition_maps_to_half_second_overlap() -> None:
    windows, _ = build_pl_speech_windows(
        [_row(1)],
        record_partitions={"record-1": "train"},
        spec=PLSpeechWindowSpec(window_sec=3.0, stride_sec=0.5, delay_ms=0),
        manifest_path="manifest.parquet",
        split_artifact_path="splits.json",
        audio_info_provider=lambda _: (12_000, 120_000),
    )
    windows = sorted(windows, key=lambda item: item.window_offset_sec)
    assert [item.window_offset_sec for item in windows] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
    ]
    assert all(item.window_sec == 3.0 for item in windows)
    assert all(item.stride_sec == 0.5 for item in windows)
    assert windows[0].audio_start_sec == 1.0
    assert windows[-1].audio_stop_sec == 7.0


def test_delay_moves_only_eeg_span_and_never_crosses_block() -> None:
    windows, _ = build_pl_speech_windows(
        [_row(1)],
        record_partitions={"record-1": "train"},
        spec=PLSpeechWindowSpec(window_sec=3.0, stride_sec=3.0, delay_ms=200),
        manifest_path="manifest.parquet",
        split_artifact_path="splits.json",
        audio_info_provider=lambda _: (12_000, 120_000),
    )
    assert len(windows) == 1
    assert windows[0].audio_start_sec == 1.0
    assert windows[0].eeg_start_sample == 1050
    assert windows[0].eeg_stop_sample <= _row(1)["eeg_end_sample"]


def test_missing_audio_and_short_rows_are_explicitly_accounted() -> None:
    rows = [_row(1, audio_file=None), _row(2, duration_sec=2.0)]
    windows, audit = build_pl_speech_windows(
        rows,
        record_partitions={"record-1": "train", "record-2": "validation"},
        spec=PLSpeechWindowSpec(),
        manifest_path="manifest.parquet",
        split_artifact_path="splits.json",
        audio_info_provider=lambda _: (12_000, 120_000),
    )
    assert windows == []
    assert audit["counts"]["excluded_record_count"] == 2
    assert set(audit["excluded_by_reason"]) == {
        "shorter_than_window_after_delay",
        "unverified_or_missing_audio_alignment",
    }


def test_window_jsonl_roundtrip_is_deterministic(tmp_path: Path) -> None:
    windows, _ = build_pl_speech_windows(
        [_row(1)],
        record_partitions={"record-1": "train"},
        spec=PLSpeechWindowSpec(),
        manifest_path="manifest.parquet",
        split_artifact_path="splits.json",
        audio_info_provider=lambda _: (12_000, 120_000),
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert write_pl_window_jsonl(first, windows) == write_pl_window_jsonl(
        second, list(reversed(windows))
    )
    assert load_pl_window_jsonl(first) == load_pl_window_jsonl(second)
