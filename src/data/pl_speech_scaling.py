"""Train-only scaling for PL EEG recordings and cached speech targets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.pl_speech import PLSpeechWindow
from features.audio_features import load_audio_sequence_features
from features.cache import safe_artifact_filename


def _training_windows(
    windows: Sequence[PLSpeechWindow],
) -> list[PLSpeechWindow]:
    selected = [window for window in windows if window.split == "train"]
    if not selected:
        raise ValueError("Scaler fitting requires at least one training window")
    return selected


@dataclass(frozen=True)
class RecordingRobustScaler:
    """Per-recording, per-channel median/IQR scaler with fixed clipping."""

    centers: Mapping[str, np.ndarray]
    scales: Mapping[str, np.ndarray]
    clamp: float = 20.0
    fitted_window_count: int = 0

    def __post_init__(self) -> None:
        if self.clamp <= 0:
            raise ValueError("clamp must be positive")
        if set(self.centers) != set(self.scales):
            raise ValueError("Robust scaler center/scale recordings differ")
        if not self.centers:
            raise ValueError("Robust scaler requires at least one recording")
        for recording, center in self.centers.items():
            scale = self.scales[recording]
            if center.ndim != 1 or scale.shape != center.shape:
                raise ValueError(
                    f"Invalid robust scaler shape for recording={recording}"
                )
            if not np.isfinite(center).all() or not np.isfinite(scale).all():
                raise ValueError(
                    f"Non-finite robust scaler state for recording={recording}"
                )
            if not (scale > 0).all():
                raise ValueError(
                    f"Non-positive robust scale for recording={recording}"
                )

    def transform(self, recording: str, eeg: np.ndarray) -> np.ndarray:
        if recording not in self.centers:
            raise KeyError(
                "No train-fitted EEG scaler for recording "
                f"{recording!r}; evaluation-time fitting is forbidden"
            )
        center = self.centers[recording][:, None]
        scale = self.scales[recording][:, None]
        if eeg.ndim != 2 or eeg.shape[0] != center.shape[0]:
            raise ValueError(
                f"EEG/scaler channel mismatch for recording={recording}: "
                f"{eeg.shape} versus {center.shape[0]} channels"
            )
        transformed = (np.asarray(eeg, dtype=np.float32) - center) / scale
        return np.asarray(
            np.clip(transformed, -self.clamp, self.clamp),
            dtype=np.float32,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "pl-recording-robust-scaler-v1",
            "centers": {
                key: torch.from_numpy(np.asarray(value, dtype=np.float32))
                for key, value in self.centers.items()
            },
            "scales": {
                key: torch.from_numpy(np.asarray(value, dtype=np.float32))
                for key, value in self.scales.items()
            },
            "clamp": self.clamp,
            "fitted_window_count": self.fitted_window_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RecordingRobustScaler":
        if state.get("schema_version") != "pl-recording-robust-scaler-v1":
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
            fitted_window_count=int(state.get("fitted_window_count", 0)),
        )

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "pl-recording-robust-scaler-v1",
            "fit_partition": "train",
            "recording_count": len(self.centers),
            "fitted_window_count": self.fitted_window_count,
            "clamp": self.clamp,
            "center": "per_recording_per_channel_median",
            "scale": "per_recording_per_channel_iqr",
            "quantile_estimator": "sorted_index_floor_q_times_n",
        }


@dataclass(frozen=True)
class SpeechStandardScaler:
    """Global scalar mean/std for speech features, matching Meta per_channel=False."""

    center: float
    scale: float
    fitted_element_count: int
    fitted_window_count: int
    fitted_unique_target_count: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.center):
            raise ValueError("Speech scaler center must be finite")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Speech scaler scale must be finite and positive")

    def transform(self, speech: np.ndarray) -> np.ndarray:
        return np.asarray(
            (np.asarray(speech, dtype=np.float32) - self.center) / self.scale,
            dtype=np.float32,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": "pl-speech-standard-scaler-v1",
            "center": self.center,
            "scale": self.scale,
            "fitted_element_count": self.fitted_element_count,
            "fitted_window_count": self.fitted_window_count,
            "fitted_unique_target_count": self.fitted_unique_target_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SpeechStandardScaler":
        if state.get("schema_version") != "pl-speech-standard-scaler-v1":
            raise ValueError("Unknown speech standard scaler state schema")
        return cls(
            center=float(state["center"]),
            scale=float(state["scale"]),
            fitted_element_count=int(state["fitted_element_count"]),
            fitted_window_count=int(state["fitted_window_count"]),
            fitted_unique_target_count=int(state["fitted_unique_target_count"]),
        )

    def audit(self) -> dict[str, object]:
        return {
            "schema_version": "pl-speech-standard-scaler-v1",
            "fit_partition": "train",
            "per_channel": False,
            "center": self.center,
            "scale": self.scale,
            "fitted_element_count": self.fitted_element_count,
            "fitted_window_count": self.fitted_window_count,
            "fitted_unique_target_count": self.fitted_unique_target_count,
            "std_correction": 1,
        }


def fit_recording_robust_scaler(
    windows: Sequence[PLSpeechWindow],
    *,
    eeg_reader: Callable[[str, int, int], np.ndarray],
    clamp: float = 20.0,
) -> RecordingRobustScaler:
    """Fit exact channel medians/IQRs from training windows only."""

    train_windows = _training_windows(windows)
    grouped: dict[str, list[PLSpeechWindow]] = defaultdict(list)
    for window in train_windows:
        grouped[window.eeg_file].append(window)

    centers: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for recording, recording_windows in sorted(grouped.items()):
        segments = [
            np.asarray(
                eeg_reader(
                    window.eeg_file,
                    window.eeg_start_sample,
                    window.eeg_stop_sample,
                ),
                dtype=np.float32,
            )
            for window in recording_windows
        ]
        channel_counts = {segment.shape[0] for segment in segments if segment.ndim == 2}
        if len(channel_counts) != 1 or any(segment.ndim != 2 for segment in segments):
            raise ValueError(
                f"Inconsistent EEG shapes while fitting recording={recording}"
            )
        samples = np.concatenate(segments, axis=1)
        ordered = np.sort(samples, axis=1)
        sample_count = ordered.shape[1]
        low = ordered[:, int(0.25 * sample_count)]
        median = ordered[:, int(0.50 * sample_count)]
        high = ordered[:, int(0.75 * sample_count)]
        scale = high - low
        scale[scale == 0] = 1.0
        centers[recording] = np.asarray(median, dtype=np.float32)
        scales[recording] = np.asarray(scale, dtype=np.float32)

    return RecordingRobustScaler(
        centers=centers,
        scales=scales,
        clamp=clamp,
        fitted_window_count=len(train_windows),
    )


def fit_speech_standard_scaler(
    windows: Sequence[PLSpeechWindow],
    *,
    feature_dir: str | Path,
    expected_model_id: str | None = None,
) -> SpeechStandardScaler:
    """Fit a scalar sample standard deviation from training targets only."""

    train_windows = _training_windows(windows)
    target_counts = Counter(window.audio_target_id for window in train_windows)
    total = np.float64(0.0)
    total_squared = np.float64(0.0)
    element_count = 0
    feature_dir = Path(feature_dir)
    for target_id, repeat_count in sorted(target_counts.items()):
        metadata, speech = load_audio_sequence_features(
            feature_dir / safe_artifact_filename(target_id)
        )
        if metadata["audio_target_id"] != target_id:
            raise ValueError(f"Speech scaler target mismatch: {target_id}")
        if metadata["split"] != "train":
            raise ValueError(f"Speech scaler received non-training target: {target_id}")
        if (
            expected_model_id is not None
            and metadata.get("model_id") != expected_model_id
        ):
            raise ValueError(
                f"Speech feature model mismatch for {target_id}: "
                f"{metadata.get('model_id')} != {expected_model_id}"
            )
        values = np.asarray(speech, dtype=np.float64)
        total += repeat_count * values.sum(dtype=np.float64)
        total_squared += repeat_count * np.square(values).sum(dtype=np.float64)
        element_count += repeat_count * values.size

    if element_count < 2:
        raise ValueError("Speech scaler requires at least two feature elements")
    center = total / element_count
    variance = (
        total_squared - (total * total) / element_count
    ) / (element_count - 1)
    variance = max(float(variance), 0.0)
    scale = variance**0.5
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Speech target features have zero or invalid variance")
    return SpeechStandardScaler(
        center=float(center),
        scale=float(scale),
        fitted_element_count=element_count,
        fitted_window_count=len(train_windows),
        fitted_unique_target_count=len(target_counts),
    )
