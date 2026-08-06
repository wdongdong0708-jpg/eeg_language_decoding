"""Guardrails for consuming official EEG derivatives without hidden reprocessing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class EEGPreprocessingPolicy:
    apply_filter: bool = False
    apply_notch: bool = False
    apply_ica: bool = False
    interpolate_bad_channels: bool = False
    rereference: bool = False
    resample: bool = False

    def validate_for_official_derivative(self) -> None:
        enabled = [
            field
            for field in (
                "apply_filter",
                "apply_notch",
                "apply_ica",
                "interpolate_bad_channels",
                "rereference",
                "resample",
            )
            if getattr(self, field)
        ]
        if enabled:
            raise ValueError(
                "Official derivatives cannot be reprocessed by default; "
                f"enabled operations: {enabled}"
            )


@dataclass(frozen=True, slots=True)
class ChannelRobustScaler:
    """Train-fitted per-recording/channel median-IQR scaling and clipping.

    This is a feature scaling operation on the released derivative, not an EEG
    reprocessing pipeline.  State is keyed by the physical EEG file so that a
    validation or test item can never silently fit its own statistics.
    """

    centers: Mapping[str, np.ndarray]
    scales: Mapping[str, np.ndarray]
    clamp: float = 5.0
    fitted_span_count: int = 0
    fitted_unique_sample_count: int = 0

    def __post_init__(self) -> None:
        if self.clamp <= 0:
            raise ValueError("clamp must be positive")
        if not self.centers or set(self.centers) != set(self.scales):
            raise ValueError("Robust scaler center/scale recordings differ")
        for recording, center in self.centers.items():
            scale = self.scales[recording]
            if center.ndim != 1 or scale.shape != center.shape:
                raise ValueError(f"Invalid robust scaler shape for {recording}")
            if not np.isfinite(center).all() or not np.isfinite(scale).all():
                raise ValueError(f"Non-finite robust scaler state for {recording}")
            if not np.all(scale > 0):
                raise ValueError(f"Non-positive robust scale for {recording}")

    def transform(self, recording: str, eeg: np.ndarray) -> np.ndarray:
        if recording not in self.centers:
            raise KeyError(
                "No train-fitted EEG scaler for recording "
                f"{recording!r}; evaluation-time fitting is forbidden"
            )
        center = np.asarray(self.centers[recording], dtype=np.float32)[:, None]
        scale = np.asarray(self.scales[recording], dtype=np.float32)[:, None]
        values = np.asarray(eeg, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != center.shape[0]:
            raise ValueError(
                f"EEG/scaler channel mismatch for recording={recording}: "
                f"{values.shape} versus {center.shape[0]} channels"
            )
        transformed = (values - center) / scale
        return np.asarray(
            np.clip(transformed, -self.clamp, self.clamp), dtype=np.float32
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "channel-robust-scaler-v1",
            "centers": {
                key: torch.from_numpy(np.asarray(value, dtype=np.float32))
                for key, value in self.centers.items()
            },
            "scales": {
                key: torch.from_numpy(np.asarray(value, dtype=np.float32))
                for key, value in self.scales.items()
            },
            "clamp": self.clamp,
            "fitted_span_count": self.fitted_span_count,
            "fitted_unique_sample_count": self.fitted_unique_sample_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ChannelRobustScaler":
        if state.get("schema_version") != "channel-robust-scaler-v1":
            raise ValueError("Unknown EEG robust scaler state schema")
        return cls(
            centers={
                str(key): np.asarray(value.cpu(), dtype=np.float32)
                for key, value in state["centers"].items()
            },
            scales={
                str(key): np.asarray(value.cpu(), dtype=np.float32)
                for key, value in state["scales"].items()
            },
            clamp=float(state["clamp"]),
            fitted_span_count=int(state.get("fitted_span_count", 0)),
            fitted_unique_sample_count=int(
                state.get("fitted_unique_sample_count", 0)
            ),
        )

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "channel-robust-scaler-v1",
            "fit_partition": "train",
            "recording_count": len(self.centers),
            "fitted_span_count": self.fitted_span_count,
            "fitted_unique_sample_count": self.fitted_unique_sample_count,
            "center": "per_recording_per_channel_median",
            "scale": "per_recording_per_channel_interquartile_range",
            "quantile_method": "numpy_linear",
            "overlap_accounting": "union_of_train_span_sample_intervals",
            "clamp": self.clamp,
        }


def _merge_sample_intervals(
    intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(stop)) for start, stop in intervals)
    if any(start < 0 or stop <= start for start, stop in ordered):
        raise ValueError("Scaler fit intervals must be ordered and non-negative")
    merged: list[tuple[int, int]] = []
    for start, stop in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            previous_start, previous_stop = merged[-1]
            merged[-1] = (previous_start, max(previous_stop, stop))
    return merged


def fit_channel_robust_scaler(
    rows: Sequence[Mapping[str, object]],
    *,
    eeg_reader: Callable[[str, int, int], np.ndarray],
    clamp: float = 5.0,
) -> ChannelRobustScaler:
    """Fit exact median/IQR statistics from the union of train span samples.

    Overlapping character windows contribute each physical EEG sample only
    once.  Passing any explicitly non-training row is rejected.
    """

    if not rows:
        raise ValueError("Robust scaler fitting requires training rows")
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        split = row.get("split")
        if split is not None and str(split) != "train":
            raise ValueError("Robust scaler received a non-training span")
        recording = str(row["eeg_file"])
        grouped[recording].append(
            (int(row["eeg_start_sample"]), int(row["eeg_stop_sample"]))
        )

    centers: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    unique_sample_count = 0
    for recording in sorted(grouped):
        intervals = _merge_sample_intervals(grouped[recording])
        segments = [
            np.asarray(eeg_reader(recording, start, stop), dtype=np.float32)
            for start, stop in intervals
        ]
        if any(segment.ndim != 2 for segment in segments):
            raise ValueError(f"Invalid EEG rank while fitting {recording}")
        channel_counts = {segment.shape[0] for segment in segments}
        if len(channel_counts) != 1:
            raise ValueError(f"Inconsistent EEG channels while fitting {recording}")
        samples = np.concatenate(segments, axis=1)
        if not np.isfinite(samples).all():
            raise ValueError(f"Non-finite EEG samples while fitting {recording}")
        low, median, high = np.quantile(
            samples,
            (0.25, 0.50, 0.75),
            axis=1,
            method="linear",
        )
        scale = high - low
        scale[scale <= np.finfo(np.float32).eps] = 1.0
        centers[recording] = np.asarray(median, dtype=np.float32)
        scales[recording] = np.asarray(scale, dtype=np.float32)
        unique_sample_count += int(samples.shape[1])

    return ChannelRobustScaler(
        centers=centers,
        scales=scales,
        clamp=clamp,
        fitted_span_count=len(rows),
        fitted_unique_sample_count=unique_sample_count,
    )
