from pathlib import Path

import numpy as np

from data.pl_speech import PL_WINDOW_SCHEMA_VERSION, PLSpeechWindow
from data.pl_speech_dataset import PLSpeechDataset
from features.audio_features import (
    AudioFeatureConfig,
    AudioFeatureInput,
    assemble_audio_frame_features,
    save_audio_sequence_features,
)
from features.cache import safe_artifact_filename


def _window() -> PLSpeechWindow:
    return PLSpeechWindow(
        window_schema_version=PL_WINDOW_SCHEMA_VERSION,
        window_id="window-1",
        record_id="record-1",
        block_id="block-1",
        split_group_id="group-1",
        split="train",
        subject_group_id="subject-group-1",
        subject_id="01",
        session_id="littleprince",
        run_id="11",
        speaker_id="f1",
        stimulus_position=1,
        char_count=12,
        eeg_file="fake.eeg",
        eeg_sampling_rate_hz=250,
        eeg_start_sample=100,
        eeg_stop_sample=850,
        eeg_sample_count=750,
        valid_eeg_samples=750,
        padded_eeg_samples=0,
        audio_file="fake.wav",
        audio_source_sample_rate_hz=12_000,
        audio_start_sample=0,
        audio_stop_sample=36_000,
        audio_start_sec=0.0,
        audio_stop_sec=3.0,
        audio_target_id="target-1",
        window_offset_sec=0.0,
        window_sec=3.0,
        stride_sec=3.0,
        eeg_delay_ms=0.0,
        source_trial_eeg_duration_sec=3.0,
        overlap_source="verified",
        quality_flag="ok",
    )


def test_dataset_loads_equal_length_eeg_and_speech(tmp_path: Path) -> None:
    window = _window()
    result = assemble_audio_frame_features(
        item=AudioFeatureInput(
            block_id=window.audio_target_id,
            content_id=window.split_group_id,
            split=window.split,
            audio_path=window.audio_file,
            start_sec=0.0,
            stop_sec=3.0,
        ),
        config=AudioFeatureConfig(model_id="fake"),
        source_sample_rate_hz=12_000,
        hidden_states=np.ones((10, 6), dtype=np.float32),
        convolution_kernels=(10, 3),
        convolution_strides=(5, 2),
    )
    save_audio_sequence_features(
        tmp_path / safe_artifact_filename(window.audio_target_id),
        audio_target_id=window.audio_target_id,
        result=result,
        target_time_steps=750,
    )
    dataset = PLSpeechDataset(
        [window],
        partition="train",
        feature_dir=tmp_path,
        eeg_reader=lambda *_: np.tile(
            np.linspace(-1, 1, 750, dtype=np.float32),
            (4, 1),
        ),
    )
    item = dataset[0]
    assert item["eeg"].shape == (4, 750)
    assert item["speech"].shape == (6, 750)
    assert item["subject_index"] == 0
    assert np.allclose(item["eeg"].mean(dim=1).numpy(), 0.0, atol=1e-6)


def test_dataset_applies_brainmagick_style_internal_offset_crop(
    tmp_path: Path,
) -> None:
    window = _window()
    result = assemble_audio_frame_features(
        item=AudioFeatureInput(
            block_id=window.audio_target_id,
            content_id=window.split_group_id,
            split=window.split,
            audio_path=window.audio_file,
            start_sec=0.0,
            stop_sec=3.0,
        ),
        config=AudioFeatureConfig(model_id="fake"),
        source_sample_rate_hz=12_000,
        hidden_states=np.ones((10, 6), dtype=np.float32),
        convolution_kernels=(10, 3),
        convolution_strides=(5, 2),
    )
    save_audio_sequence_features(
        tmp_path / safe_artifact_filename(window.audio_target_id),
        audio_target_id=window.audio_target_id,
        result=result,
        target_time_steps=750,
    )
    raw_eeg = np.tile(
        np.linspace(-1, 1, 750, dtype=np.float32),
        (4, 1),
    )
    dataset = PLSpeechDataset(
        [window],
        partition="train",
        feature_dir=tmp_path,
        alignment_offset_ms=500,
        eeg_reader=lambda *_: raw_eeg,
        cache_speech_targets=True,
    )
    item = dataset[0]
    assert dataset.alignment_offset_samples == 125
    assert item["eeg"].shape == (4, 625)
