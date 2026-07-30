"""Frame-level wav2vec hidden states with explicit local time coordinates."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data.manifest import VALID_SPLITS

OFFICIAL_AUDIO_MODEL = "airesearch/wav2vec2-large-xlsr-53-th"
OFFICIAL_AUDIO_POOLING = "last_hidden_state.mean(dim=1)"
MODEL_SAMPLE_RATE_HZ = 16_000


@dataclass(frozen=True, slots=True)
class AudioFeatureInput:
    block_id: str
    content_id: str
    split: str
    audio_path: str
    start_sec: float
    stop_sec: float

    def validate(self) -> None:
        if not self.block_id or not self.content_id or not self.split:
            raise ValueError("block_id, content_id and inherited split are required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown inherited split: {self.split!r}")
        if self.start_sec < 0 or self.stop_sec <= self.start_sec:
            raise ValueError("Audio block boundaries must be positive and ordered")


@dataclass(frozen=True, slots=True)
class AudioFeatureConfig:
    model_id: str = OFFICIAL_AUDIO_MODEL
    target_sample_rate_hz: int = MODEL_SAMPLE_RATE_HZ
    layer_index: int = -1
    output_dtype: str = "float32"

    def validate(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        np.dtype(self.output_dtype)


@dataclass(frozen=True, slots=True)
class AudioFrameFeatureResult:
    block_id: str
    content_id: str
    split: str
    audio_path: str
    model_id: str
    layer_index: int
    source_sample_rate_hz: int
    model_sample_rate_hz: int
    source_start_sample: int
    source_stop_sample: int
    source_start_sec: float
    source_stop_sec: float
    frame_indices: np.ndarray
    frame_offsets_sec: np.ndarray
    frame_hidden_states: np.ndarray
    convolution_stride_samples: int
    receptive_field_samples: int

    def validate(self) -> None:
        frame_count = self.frame_hidden_states.shape[0]
        if self.frame_hidden_states.ndim != 2:
            raise ValueError("frame_hidden_states must have shape [frames, hidden]")
        if self.frame_indices.shape != (frame_count,):
            raise ValueError("frame_indices shape is inconsistent")
        if self.frame_offsets_sec.shape != (frame_count, 2):
            raise ValueError("frame_offsets_sec must have shape [frames, 2]")
        if frame_count and np.any(
            self.frame_offsets_sec[:, 1] <= self.frame_offsets_sec[:, 0]
        ):
            raise ValueError("Every audio frame must have a positive time span")
        if self.source_stop_sec <= self.source_start_sec:
            raise ValueError("Source audio boundaries are invalid")
        if (
            self.source_start_sample < 0
            or self.source_stop_sample <= self.source_start_sample
        ):
            raise ValueError("Source audio sample boundaries are invalid")
        if frame_count and (
            self.frame_offsets_sec[:, 0].min() < self.source_start_sec
            or self.frame_offsets_sec[:, 1].max() > self.source_stop_sec
        ):
            raise ValueError("Audio frame support crosses the source block boundary")


def convolution_geometry(
    kernels: Sequence[int],
    strides: Sequence[int],
) -> tuple[int, int]:
    """Return output stride and receptive field in input waveform samples."""

    if len(kernels) != len(strides) or not kernels:
        raise ValueError("Convolution kernels and strides must have equal non-zero length")
    jump = 1
    receptive_field = 1
    for kernel, stride in zip(kernels, strides, strict=True):
        if kernel <= 0 or stride <= 0:
            raise ValueError("Convolution kernels and strides must be positive")
        receptive_field += (kernel - 1) * jump
        jump *= stride
    return jump, receptive_field


def frame_time_offsets(
    *,
    frame_count: int,
    sample_rate_hz: int,
    stride_samples: int,
    receptive_field_samples: int,
    source_start_sec: float = 0.0,
    source_stop_sec: float | None = None,
) -> np.ndarray:
    """Compute receptive-field bounds for every wav2vec frame."""

    if frame_count < 0:
        raise ValueError("frame_count cannot be negative")
    if sample_rate_hz <= 0 or stride_samples <= 0 or receptive_field_samples <= 0:
        raise ValueError("Sample rate and convolution geometry must be positive")
    starts = (
        source_start_sec
        + np.arange(frame_count, dtype=np.float64) * stride_samples / sample_rate_hz
    )
    stops = starts + receptive_field_samples / sample_rate_hz
    if source_stop_sec is not None:
        stops = np.minimum(stops, source_stop_sec)
    return np.column_stack([starts, stops]).reshape(frame_count, 2)


def assemble_audio_frame_features(
    *,
    item: AudioFeatureInput,
    config: AudioFeatureConfig,
    source_sample_rate_hz: int,
    hidden_states: np.ndarray,
    convolution_kernels: Sequence[int],
    convolution_strides: Sequence[int],
) -> AudioFrameFeatureResult:
    """Attach source-time coordinates and provenance to frame hidden states."""

    item.validate()
    config.validate()
    hidden = np.asarray(hidden_states)
    if hidden.ndim != 2:
        raise ValueError("hidden_states must have shape [frames, hidden]")
    stride_samples, receptive_samples = convolution_geometry(
        convolution_kernels,
        convolution_strides,
    )
    source_start_sample = round(item.start_sec * source_sample_rate_hz)
    source_stop_sample = round(item.stop_sec * source_sample_rate_hz)
    if source_stop_sample <= source_start_sample:
        raise ValueError("Audio block is empty after source-sample quantization")
    source_start_sec = source_start_sample / source_sample_rate_hz
    source_stop_sec = source_stop_sample / source_sample_rate_hz
    offsets = frame_time_offsets(
        frame_count=hidden.shape[0],
        sample_rate_hz=config.target_sample_rate_hz,
        stride_samples=stride_samples,
        receptive_field_samples=receptive_samples,
        source_start_sec=source_start_sec,
        source_stop_sec=source_stop_sec,
    )
    result = AudioFrameFeatureResult(
        block_id=item.block_id,
        content_id=item.content_id,
        split=item.split,
        audio_path=item.audio_path,
        model_id=config.model_id,
        layer_index=config.layer_index,
        source_sample_rate_hz=source_sample_rate_hz,
        model_sample_rate_hz=config.target_sample_rate_hz,
        source_start_sample=source_start_sample,
        source_stop_sample=source_stop_sample,
        source_start_sec=source_start_sec,
        source_stop_sec=source_stop_sec,
        frame_indices=np.arange(hidden.shape[0], dtype=np.int64),
        frame_offsets_sec=offsets,
        frame_hidden_states=np.asarray(hidden, dtype=np.dtype(config.output_dtype)),
        convolution_stride_samples=stride_samples,
        receptive_field_samples=receptive_samples,
    )
    result.validate()
    return result


def load_audio_block(
    item: AudioFeatureInput,
) -> tuple[np.ndarray, int]:
    """Load exactly one verified block and mix channels to mono."""

    import soundfile as sf

    item.validate()
    path = Path(item.audio_path)
    info = sf.info(path)
    start_frame = round(item.start_sec * info.samplerate)
    stop_frame = round(item.stop_sec * info.samplerate)
    if stop_frame <= start_frame:
        raise ValueError("Audio block is empty after source-sample quantization")
    if stop_frame > info.frames:
        raise ValueError(
            f"Audio block exceeds file duration: stop={item.stop_sec}, "
            f"duration={info.frames / info.samplerate}"
        )
    waveform, sample_rate = sf.read(
        path,
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=True,
    )
    mono = waveform.mean(axis=1, dtype=np.float32)
    return mono, int(sample_rate)


class Wav2VecFrameExtractor:
    """Extract unpooled wav2vec hidden states for a single verified audio block."""

    def __init__(
        self,
        *,
        processor: object,
        model: object,
        config: AudioFeatureConfig,
        device: str | None = None,
    ) -> None:
        config.validate()
        self.processor = processor
        self.model = model
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        config: AudioFeatureConfig,
        *,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> "Wav2VecFrameExtractor":
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            config.model_id,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            config.model_id,
            local_files_only=local_files_only,
        )
        if device is not None:
            model = model.to(device)
        model.eval()
        return cls(processor=processor, model=model, config=config, device=device)

    def extract(self, item: AudioFeatureInput) -> AudioFrameFeatureResult:
        import torch
        import torchaudio.functional as audio_functional

        waveform, source_rate = load_audio_block(item)
        waveform_tensor = torch.from_numpy(waveform)
        if source_rate != self.config.target_sample_rate_hz:
            waveform_tensor = audio_functional.resample(
                waveform_tensor,
                source_rate,
                self.config.target_sample_rate_hz,
            )
        processed = self.processor(
            waveform_tensor.numpy(),
            sampling_rate=self.config.target_sample_rate_hz,
            return_tensors="pt",
        )
        model_inputs = {
            name: tensor.to(self.device) if self.device else tensor
            for name, tensor in processed.items()
        }
        needs_all_layers = self.config.layer_index != -1
        with torch.inference_mode():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=needs_all_layers,
            )
        hidden = (
            outputs.last_hidden_state
            if self.config.layer_index == -1
            else outputs.hidden_states[self.config.layer_index]
        )
        hidden_np = hidden[0].detach().cpu().numpy()
        kernels = tuple(int(value) for value in self.model.config.conv_kernel)
        strides = tuple(int(value) for value in self.model.config.conv_stride)
        return assemble_audio_frame_features(
            item=item,
            config=self.config,
            source_sample_rate_hz=source_rate,
            hidden_states=hidden_np,
            convolution_kernels=kernels,
            convolution_strides=strides,
        )


def save_audio_frame_features(
    path: str | Path,
    result: AudioFrameFeatureResult,
) -> None:
    result.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "block_id": result.block_id,
        "content_id": result.content_id,
        "split": result.split,
        "audio_path": result.audio_path,
        "model_id": result.model_id,
        "layer_index": result.layer_index,
        "source_sample_rate_hz": result.source_sample_rate_hz,
        "model_sample_rate_hz": result.model_sample_rate_hz,
        "source_start_sample": result.source_start_sample,
        "source_stop_sample": result.source_stop_sample,
        "source_start_sec": result.source_start_sec,
        "source_stop_sec": result.source_stop_sec,
        "convolution_stride_samples": result.convolution_stride_samples,
        "receptive_field_samples": result.receptive_field_samples,
    }
    handle = tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=target.stem + ".",
        suffix=".npz",
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            frame_indices=result.frame_indices,
            frame_offsets_sec=result.frame_offsets_sec,
            frame_hidden_states=result.frame_hidden_states,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_audio_frame_features(path: str | Path) -> AudioFrameFeatureResult:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        result = AudioFrameFeatureResult(
            block_id=metadata["block_id"],
            content_id=metadata["content_id"],
            split=metadata["split"],
            audio_path=metadata["audio_path"],
            model_id=metadata["model_id"],
            layer_index=int(metadata["layer_index"]),
            source_sample_rate_hz=int(metadata["source_sample_rate_hz"]),
            model_sample_rate_hz=int(metadata["model_sample_rate_hz"]),
            source_start_sample=int(metadata["source_start_sample"]),
            source_stop_sample=int(metadata["source_stop_sample"]),
            source_start_sec=float(metadata["source_start_sec"]),
            source_stop_sec=float(metadata["source_stop_sec"]),
            frame_indices=archive["frame_indices"],
            frame_offsets_sec=archive["frame_offsets_sec"],
            frame_hidden_states=archive["frame_hidden_states"],
            convolution_stride_samples=int(
                metadata["convolution_stride_samples"]
            ),
            receptive_field_samples=int(metadata["receptive_field_samples"]),
        )
    result.validate()
    return result


def mean_pool_audio_frames(
    result: AudioFrameFeatureResult,
    *,
    start_sec: float,
    stop_sec: float,
) -> np.ndarray:
    """Pool frames overlapping a requested span without discarding frame output."""

    if start_sec < result.source_start_sec or stop_sec > result.source_stop_sec:
        raise ValueError("Pooling span must remain inside the source block")
    selected = (
        (result.frame_offsets_sec[:, 0] < stop_sec)
        & (result.frame_offsets_sec[:, 1] > start_sec)
    )
    if not selected.any():
        raise ValueError("No wav2vec frames overlap the requested span")
    return result.frame_hidden_states[selected].mean(axis=0)
