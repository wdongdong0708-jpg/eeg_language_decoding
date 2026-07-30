from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

from data.pl_speech import load_pl_window_jsonl


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "all_trials.parquet"
WINDOWS = ROOT / "metadata" / "pl_speech_windows_seed42_3s_delay_000ms.jsonl"
AUDIT = ROOT / "reports" / "pl_speech_windows_seed42_3s_delay_000ms.json"

pytestmark = pytest.mark.integration


def _require_artifacts() -> None:
    if not MANIFEST.exists():
        pytest.skip("Local real manifest is unavailable")
    assert WINDOWS.is_file()
    assert AUDIT.is_file()


def test_real_pl_windows_are_exact_unpadded_and_protocol_disjoint() -> None:
    _require_artifacts()
    windows = load_pl_window_jsonl(WINDOWS)
    assert len(windows) == 2_677
    assert all(window.eeg_sample_count == 750 for window in windows)
    assert all(window.valid_eeg_samples == 750 for window in windows)
    assert all(window.padded_eeg_samples == 0 for window in windows)
    assert all(window.window_sec == 3.0 for window in windows)
    assert all(window.stride_sec == 3.0 for window in windows)
    assert all(window.eeg_delay_ms == 0.0 for window in windows)
    group_splits: dict[str, set[str]] = defaultdict(set)
    target_splits: dict[str, set[str]] = defaultdict(set)
    for window in windows:
        group_splits[window.split_group_id].add(window.split)
        target_splits[window.audio_target_id].add(window.split)
    assert all(len(values) == 1 for values in group_splits.values())
    assert all(len(values) == 1 for values in target_splits.values())


def test_real_pl_window_accounting_and_sha_are_complete() -> None:
    _require_artifacts()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["counts"] == {
        "audio_target_count": 688,
        "content_group_count": 346,
        "excluded_record_count": 37_625,
        "input_pl_record_count": 40_302,
        "records_with_windows": 2_677,
        "subject_group_count": 8,
        "window_count": 2_677,
        "window_counts_by_partition": {
            "test": 295,
            "train": 2_173,
            "validation": 209,
        },
    }
    assert {
        reason: len(record_ids)
        for reason, record_ids in audit["excluded_by_reason"].items()
    } == {
        "audio_bounds_exceed_file": 179,
        "shorter_than_window_after_delay": 18_924,
        "unverified_or_missing_audio_alignment": 18_522,
    }
    assert audit["leakage_checks"] == {
        "all_windows_exact_length": True,
        "audio_target_cross_partition_count": 0,
        "content_group_cross_partition_count": 0,
        "duplicate_window_id_count": 0,
        "record_level_selected_plus_excluded_equals_input": True,
    }
    assert (
        hashlib.sha256(WINDOWS.read_bytes()).hexdigest()
        == audit["window_jsonl_sha256"]
    )
