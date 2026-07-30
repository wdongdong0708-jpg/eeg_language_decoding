import numpy as np
import torch

from evaluation.pl_shortcuts import (
    correlation_scores,
    eeg_global_field_power,
    exact_metadata_scores,
    membership_metadata_scores,
    smoothed_audio_envelope,
)


def test_exact_metadata_scores_expose_ties() -> None:
    scores = exact_metadata_scores(["a", "b"], ["a", "a", "b"])
    assert torch.equal(
        scores,
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    membership = membership_metadata_scores(
        ["a", "b"],
        [{"a", "b"}, {"b"}],
    )
    assert torch.equal(
        membership,
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
    )


def test_audio_and_eeg_envelopes_have_requested_time_length() -> None:
    time = np.linspace(0, 1, 12_000, endpoint=False)
    waveform = np.sin(2 * np.pi * 3 * time).astype(np.float32)
    eeg = np.tile(waveform[::48], (4, 1))
    audio_envelope = smoothed_audio_envelope(
        waveform,
        sample_rate_hz=12_000,
        target_time_steps=250,
    )
    eeg_envelope = eeg_global_field_power(
        eeg,
        sample_rate_hz=250,
        target_time_steps=250,
    )
    assert audio_envelope.shape == (250,)
    assert eeg_envelope.shape == (250,)
    scores = correlation_scores(
        eeg_envelope[None],
        audio_envelope[None],
    )
    assert scores.shape == (1, 1)
