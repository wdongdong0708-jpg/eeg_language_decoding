from pathlib import Path

import numpy as np
import pytest
import torch

from features.audio_features import (
    AudioFeatureConfig,
    AudioFeatureInput,
    average_hidden_layers,
    assemble_audio_frame_features,
    convolution_geometry,
    interpolate_audio_sequence,
    load_audio_frame_features,
    load_audio_sequence_features,
    save_audio_sequence_features,
    save_audio_frame_features,
)


def _result():
    item = AudioFeatureInput(
        block_id="block-1",
        content_id="content-1",
        split="test",
        audio_path="audio.wav",
        start_sec=10.0,
        stop_sec=10.1,
    )
    return assemble_audio_frame_features(
        item=item,
        config=AudioFeatureConfig(model_id="fake-wav2vec"),
        source_sample_rate_hz=12_000,
        hidden_states=np.arange(8, dtype=np.float32).reshape(4, 2),
        convolution_kernels=(10, 3, 3, 3, 3, 2, 2),
        convolution_strides=(5, 2, 2, 2, 2, 2, 2),
    )


def test_wav2vec_convolution_geometry_and_frame_times() -> None:
    stride, receptive = convolution_geometry(
        (10, 3, 3, 3, 3, 2, 2),
        (5, 2, 2, 2, 2, 2, 2),
    )
    assert stride == 320
    assert receptive == 400
    result = _result()
    assert result.frame_offsets_sec[:, 0] == pytest.approx(
        [10.0, 10.02, 10.04, 10.06]
    )
    assert result.frame_offsets_sec[:, 1] == pytest.approx(
        [10.025, 10.045, 10.065, 10.085]
    )
    assert result.source_start_sample == 120_000
    assert result.source_stop_sample == 121_200


def test_audio_feature_npz_roundtrip_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "audio.npz"
    original = _result()
    save_audio_frame_features(path, original)
    restored = load_audio_frame_features(path)
    assert restored.split == "test"
    assert restored.content_id == "content-1"
    assert np.array_equal(restored.frame_hidden_states, original.frame_hidden_states)


def test_audio_feature_input_rejects_unknown_split() -> None:
    item = AudioFeatureInput(
        block_id="block-1",
        content_id="content-1",
        split="fold-1",
        audio_path="audio.wav",
        start_sec=0.0,
        stop_sec=1.0,
    )
    with pytest.raises(ValueError, match="Unknown inherited split"):
        item.validate()


def test_selected_layers_are_averaged_without_time_pooling() -> None:
    hidden_states = tuple(
        torch.full((1, 5, 3), float(index)) for index in range(20)
    )
    averaged = average_hidden_layers(hidden_states, (14, 15, 16, 17, 18))
    assert averaged.shape == (1, 5, 3)
    assert torch.all(averaged == 16.0)


def test_audio_sequence_interpolation_preserves_channels() -> None:
    frames = np.arange(12, dtype=np.float32).reshape(4, 3)
    sequence = interpolate_audio_sequence(frames, target_time_steps=10)
    assert sequence.shape == (3, 10)


def test_audio_sequence_npz_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sequence.npz"
    save_audio_sequence_features(
        path,
        audio_target_id="target-1",
        result=_result(),
        target_time_steps=25,
    )
    metadata, features = load_audio_sequence_features(path)
    assert metadata["audio_target_id"] == "target-1"
    assert metadata["temporal_pooling"] is False
    assert features.shape == (2, 25)
