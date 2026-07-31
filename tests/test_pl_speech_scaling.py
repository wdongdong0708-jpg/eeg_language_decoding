import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data.pl_speech import PL_WINDOW_SCHEMA_VERSION, PLSpeechWindow
from data.pl_speech_scaling import (
    RecordingRobustScaler,
    SpeechStandardScaler,
    fit_recording_robust_scaler,
    fit_speech_standard_scaler,
)
from features.cache import safe_artifact_filename


def _window(*, split: str, target_id: str, start: int = 0) -> PLSpeechWindow:
    return PLSpeechWindow(
        window_schema_version=PL_WINDOW_SCHEMA_VERSION,
        window_id=f"{split}-{target_id}",
        record_id=f"record-{target_id}",
        block_id=f"block-{target_id}",
        split_group_id=f"group-{target_id}",
        split=split,
        subject_group_id="subject-group-1",
        subject_id="01",
        session_id="littleprince",
        run_id="11",
        speaker_id="f1",
        stimulus_position=1,
        char_count=12,
        eeg_file="recording.eeg",
        eeg_sampling_rate_hz=2,
        eeg_start_sample=start,
        eeg_stop_sample=start + 2,
        eeg_sample_count=2,
        valid_eeg_samples=2,
        padded_eeg_samples=0,
        audio_file="fake.wav",
        audio_source_sample_rate_hz=16_000,
        audio_start_sample=0,
        audio_stop_sample=32_000,
        audio_start_sec=0.0,
        audio_stop_sec=1.0,
        audio_target_id=target_id,
        window_offset_sec=0.0,
        window_sec=1.0,
        stride_sec=1.0,
        eeg_delay_ms=0.0,
        source_trial_eeg_duration_sec=1.0,
        overlap_source="verified",
        quality_flag="ok",
    )


def _save_feature(
    directory: Path,
    *,
    target_id: str,
    split: str,
    values: np.ndarray,
) -> None:
    metadata = {
        "schema_version": "pl-audio-sequence-v1",
        "audio_target_id": target_id,
        "split": split,
        "model_id": "facebook/wav2vec2-large-xlsr-53",
        "feature_shape": list(values.shape),
        "target_time_steps": values.shape[1],
    }
    np.savez_compressed(
        directory / safe_artifact_filename(target_id),
        metadata_json=np.asarray(json.dumps(metadata)),
        sequence_features=np.asarray(values, dtype=np.float32),
    )


def test_recording_robust_scaler_uses_train_only_and_clamps() -> None:
    train = _window(split="train", target_id="train", start=0)
    validation = replace(
        _window(split="validation", target_id="validation", start=2),
        eeg_file=train.eeg_file,
    )
    segments = {
        0: np.asarray([[0.0, 2.0], [10.0, 14.0]], dtype=np.float32),
        2: np.full((2, 2), 10_000.0, dtype=np.float32),
    }
    scaler = fit_recording_robust_scaler(
        [train, validation],
        eeg_reader=lambda _file, start, _stop: segments[start],
        clamp=20.0,
    )
    assert scaler.fitted_window_count == 1
    assert scaler.centers[train.eeg_file] == pytest.approx([2.0, 14.0])
    restored = RecordingRobustScaler.from_state_dict(scaler.state_dict())
    transformed = restored.transform(
        train.eeg_file,
        np.asarray([[2.0, 1_000.0], [14.0, -1_000.0]], dtype=np.float32),
    )
    assert transformed[:, 0] == pytest.approx([0.0, 0.0])
    assert transformed[:, 1] == pytest.approx([20.0, -20.0])


def test_speech_standard_scaler_is_global_and_train_only(tmp_path: Path) -> None:
    train = _window(split="train", target_id="train")
    validation = _window(split="validation", target_id="validation")
    _save_feature(
        tmp_path,
        target_id="train",
        split="train",
        values=np.asarray([[0.0, 2.0]], dtype=np.float32),
    )
    _save_feature(
        tmp_path,
        target_id="validation",
        split="validation",
        values=np.asarray([[1000.0, 1000.0]], dtype=np.float32),
    )
    scaler = fit_speech_standard_scaler(
        [train, validation],
        feature_dir=tmp_path,
        expected_model_id="facebook/wav2vec2-large-xlsr-53",
    )
    assert scaler.center == pytest.approx(1.0)
    assert scaler.scale == pytest.approx(2.0**0.5)
    assert scaler.fitted_unique_target_count == 1
    restored = SpeechStandardScaler.from_state_dict(scaler.state_dict())
    assert restored.transform(np.asarray([[1.0]])).item() == pytest.approx(0.0)
