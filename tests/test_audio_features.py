from pathlib import Path

import numpy as np
import pytest

from features.audio_features import (
    AudioFeatureConfig,
    AudioFeatureInput,
    assemble_audio_frame_features,
    convolution_geometry,
    load_audio_frame_features,
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
