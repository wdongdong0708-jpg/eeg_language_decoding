"""Shortcut-only retrieval scores for fixed-duration PL windows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from evaluation.retrieval_metrics import RetrievalMetrics, metrics_from_ranks
from evaluation.speech_retrieval import ranks_from_score_matrix


def random_baseline_metrics(
    *,
    query_count: int,
    candidate_count: int,
    seed: int,
) -> RetrievalMetrics:
    if query_count < 1 or candidate_count < 2:
        raise ValueError("random retrieval requires queries and at least two candidates")
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(query_count, candidate_count, generator=generator)
    positives = torch.arange(query_count) % candidate_count
    return metrics_from_ranks(
        ranks_from_score_matrix(
            scores,
            positives,
            tie_policy="pessimistic",
        )
    )


def exact_metadata_scores(
    query_values: Sequence[object],
    candidate_values: Sequence[object],
) -> torch.Tensor:
    """Score one for exact metadata equality and zero otherwise."""

    return torch.tensor(
        [
            [float(query == candidate) for candidate in candidate_values]
            for query in query_values
        ],
        dtype=torch.float32,
    )


def membership_metadata_scores(
    query_values: Sequence[object],
    candidate_value_sets: Sequence[set[object]],
) -> torch.Tensor:
    """Score membership when one audio target has several observed metadata values."""

    return torch.tensor(
        [
            [float(query in values) for values in candidate_value_sets]
            for query in query_values
        ],
        dtype=torch.float32,
    )


def smoothed_audio_envelope(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    target_time_steps: int,
    smoothing_ms: float = 50.0,
) -> np.ndarray:
    mono = np.asarray(waveform, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if mono.ndim != 1 or mono.size < 2:
        raise ValueError("waveform must contain a non-empty mono/time signal")
    squared = np.square(mono, dtype=np.float32)
    kernel_size = max(1, round(smoothing_ms / 1000.0 * sample_rate_hz))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    rms = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return _interpolate_and_standardize(rms, target_time_steps)


def eeg_global_field_power(
    eeg: np.ndarray,
    *,
    sample_rate_hz: int,
    target_time_steps: int,
    smoothing_ms: float = 50.0,
) -> np.ndarray:
    signal = np.asarray(eeg, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] < 2:
        raise ValueError("eeg must have shape [channels, time]")
    channel_mean = signal.mean(axis=1, keepdims=True)
    channel_std = signal.std(axis=1, keepdims=True)
    standardized = (signal - channel_mean) / np.maximum(channel_std, 1e-8)
    gfp = np.sqrt(np.mean(np.square(standardized), axis=0))
    kernel_size = max(1, round(smoothing_ms / 1000.0 * sample_rate_hz))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(gfp, kernel, mode="same")
    return _interpolate_and_standardize(smoothed, target_time_steps)


def correlation_scores(
    query_sequences: np.ndarray,
    candidate_sequences: np.ndarray,
) -> torch.Tensor:
    queries = torch.as_tensor(query_sequences, dtype=torch.float32)
    candidates = torch.as_tensor(candidate_sequences, dtype=torch.float32)
    if queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError("shortcut sequences must have shape [items, time]")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("query and candidate sequences need equal time length")
    queries = torch.nn.functional.normalize(queries, dim=1)
    candidates = torch.nn.functional.normalize(candidates, dim=1)
    return queries @ candidates.T


def _interpolate_and_standardize(
    values: np.ndarray,
    target_time_steps: int,
) -> np.ndarray:
    if target_time_steps < 2:
        raise ValueError("target_time_steps must be at least two")
    source_positions = np.linspace(0.0, 1.0, len(values), endpoint=True)
    target_positions = np.linspace(
        0.0,
        1.0,
        target_time_steps,
        endpoint=True,
    )
    output = np.interp(target_positions, source_positions, values)
    output = output - output.mean()
    scale = output.std()
    if scale > 1e-8:
        output = output / scale
    return np.asarray(output, dtype=np.float32)
