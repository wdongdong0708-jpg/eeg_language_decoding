"""PyTorch dataset for synchronized PL EEG and cached speech sequences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.pl_speech import PLSpeechWindow, load_pl_window_jsonl
from data.pl_speech_scaling import RecordingRobustScaler, SpeechStandardScaler
from features.audio_features import load_audio_sequence_features
from features.cache import safe_artifact_filename

EEGNormalization = Literal[
    "none",
    "per_window_channel_zscore",
    "train_recording_robust_clamp",
]


class BrainVisionSegmentReader:
    """Lazy, per-process BrainVision reader for official derivatives."""

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def __call__(
        self,
        eeg_file: str,
        start_sample: int,
        stop_sample: int,
    ) -> np.ndarray:
        import mne

        header = str(Path(eeg_file).with_suffix(".vhdr"))
        raw = self._cache.get(header)
        if raw is None:
            if not Path(header).is_file():
                raise FileNotFoundError(f"BrainVision header not found: {header}")
            raw = mne.io.read_raw_brainvision(
                header,
                preload=False,
                verbose="ERROR",
            )
            self._cache[header] = raw
        return np.asarray(
            raw.get_data(start=start_sample, stop=stop_sample),
            dtype=np.float32,
        )


class PLSpeechDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        windows: Sequence[PLSpeechWindow] | str | Path,
        *,
        partition: str,
        feature_dir: str | Path,
        eeg_normalization: EEGNormalization = "per_window_channel_zscore",
        eeg_scaler: RecordingRobustScaler | None = None,
        speech_scaler: SpeechStandardScaler | None = None,
        expected_audio_model_id: str | None = None,
        eeg_reader: Callable[[str, int, int], np.ndarray] | None = None,
        require_all_features: bool = True,
        cache_speech_targets: bool = False,
    ) -> None:
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown partition: {partition}")
        source = (
            load_pl_window_jsonl(windows)
            if isinstance(windows, (str, Path))
            else list(windows)
        )
        self.windows = sorted(
            (window for window in source if window.split == partition),
            key=lambda window: (
                window.audio_target_id,
                window.subject_group_id,
                window.record_id,
                window.window_offset_sec,
            ),
        )
        if not self.windows:
            raise ValueError(f"No PL speech windows for partition={partition}")
        if eeg_normalization not in {
            "none",
            "per_window_channel_zscore",
            "train_recording_robust_clamp",
        }:
            raise ValueError(f"Unknown EEG normalization: {eeg_normalization}")
        if (
            eeg_normalization == "train_recording_robust_clamp"
            and eeg_scaler is None
        ):
            raise ValueError(
                "train_recording_robust_clamp requires a train-fitted EEG scaler"
            )
        self.partition = partition
        self.feature_dir = Path(feature_dir)
        self.eeg_normalization = eeg_normalization
        self.eeg_scaler = eeg_scaler
        self.speech_scaler = speech_scaler
        self.expected_audio_model_id = expected_audio_model_id
        self.eeg_reader = eeg_reader or BrainVisionSegmentReader()
        self.cache_speech_targets = cache_speech_targets
        self._speech_target_cache: dict[str, torch.Tensor] = {}
        self.audio_target_ids = [
            window.audio_target_id for window in self.windows
        ]
        if require_all_features:
            missing = [
                window.audio_target_id
                for window in self.windows
                if not self._feature_path(window.audio_target_id).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing {len(set(missing))} speech feature files; "
                    f"first target={missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        eeg = self.eeg_reader(
            window.eeg_file,
            window.eeg_start_sample,
            window.eeg_stop_sample,
        )
        if eeg.ndim != 2 or eeg.shape[1] != window.eeg_sample_count:
            raise ValueError(
                f"EEG segment shape mismatch for {window.window_id}: {eeg.shape}"
            )
        if self.eeg_normalization == "per_window_channel_zscore":
            eeg = _per_window_channel_zscore(eeg)
        elif self.eeg_normalization == "train_recording_robust_clamp":
            assert self.eeg_scaler is not None
            eeg = self.eeg_scaler.transform(window.eeg_file, eeg)
        if self.cache_speech_targets:
            speech_tensor = self.load_speech_target(window.audio_target_id)
            speech = speech_tensor.numpy()
        else:
            metadata, speech = load_audio_sequence_features(
                self._feature_path(window.audio_target_id)
            )
            if metadata["audio_target_id"] != window.audio_target_id:
                raise ValueError(
                    f"Speech target ID mismatch for {window.window_id}"
                )
            if metadata["split"] != window.split:
                raise ValueError(f"Speech target changed split for {window.window_id}")
            self._validate_audio_model(metadata, window.audio_target_id)
            if self.speech_scaler is not None:
                speech = self.speech_scaler.transform(speech)
        if speech.shape[1] != eeg.shape[1]:
            raise ValueError(
                f"EEG/speech time mismatch for {window.window_id}: "
                f"{eeg.shape[1]} != {speech.shape[1]}"
            )
        return {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "speech": (
                speech_tensor
                if self.cache_speech_targets
                else torch.from_numpy(np.ascontiguousarray(speech))
            ),
            "window_id": window.window_id,
            "audio_target_id": window.audio_target_id,
            "split_group_id": window.split_group_id,
            "subject_group_id": window.subject_group_id,
            "stimulus_position": window.stimulus_position,
        }

    def load_speech_target(self, audio_target_id: str) -> torch.Tensor:
        cached = self._speech_target_cache.get(audio_target_id)
        if cached is not None:
            return cached
        metadata, speech = load_audio_sequence_features(
            self._feature_path(audio_target_id)
        )
        if metadata["audio_target_id"] != audio_target_id:
            raise ValueError(f"Speech cache target mismatch: {audio_target_id}")
        if metadata["split"] != self.partition:
            raise ValueError(f"Speech cache partition mismatch: {audio_target_id}")
        self._validate_audio_model(metadata, audio_target_id)
        if self.speech_scaler is not None:
            speech = self.speech_scaler.transform(speech)
        tensor = torch.from_numpy(np.ascontiguousarray(speech))
        self._speech_target_cache[audio_target_id] = tensor
        return tensor

    def _feature_path(self, audio_target_id: str) -> Path:
        return self.feature_dir / safe_artifact_filename(audio_target_id)

    def _validate_audio_model(
        self,
        metadata: dict[str, object],
        audio_target_id: str,
    ) -> None:
        if (
            self.expected_audio_model_id is not None
            and metadata.get("model_id") != self.expected_audio_model_id
        ):
            raise ValueError(
                f"Speech feature model mismatch for {audio_target_id}: "
                f"{metadata.get('model_id')} != {self.expected_audio_model_id}"
            )


def _per_window_channel_zscore(
    eeg: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    mean = eeg.mean(axis=1, keepdims=True, dtype=np.float64)
    std = eeg.std(axis=1, keepdims=True, dtype=np.float64)
    normalized = (eeg - mean) / np.maximum(std, eps)
    return np.asarray(normalized, dtype=np.float32)
