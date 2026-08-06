"""PyTorch dataset for fixed-length ChineseEEG1 EEG/text spans."""

from __future__ import annotations

import configparser
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.chineseeeg1_timeline import chineseeeg1_clock_positions
from features.cache import safe_artifact_filename
from features.text_features import (
    TextFeatureResult,
    load_text_features,
    pool_text_span,
    select_text_span_by_offsets,
)
from preprocessing.eeg import ChannelRobustScaler

EEGNormalization = Literal[
    "none",
    "per_span_channel_zscore",
    "train_recording_robust_clamp",
]
TextTargetMode = Literal[
    "local_mean",
    "local_cls",
    "local_character_sequence",
    "sentence_span_mean",
    "sentence_cls",
    "sentence_character_sequence",
]

DATASET_COLUMNS = [
    "record_id",
    "span_event_id",
    "span_text_id",
    "global_text_id",
    "split_group_id",
    "split",
    "subject_group_id",
    "subject_id",
    "session_id",
    "book_id",
    "run_id",
    "stimulus_position",
    "source_sentence_text",
    "span_text",
    "span_char_count",
    "span_start_char",
    "span_end_char",
    "span_start_clock",
    "span_end_clock",
    "eeg_file",
    "eeg_start_sample",
    "eeg_stop_sample",
    "source_eeg_sample_count",
    "model_eeg_sample_count",
    "padding_samples",
    "exposes_padding_mask",
]

OPTIONAL_DATASET_COLUMNS = [
    "is_semantic_unit",
    "semantic_unit_kind",
    "semantic_pos_pattern",
    "semantic_unit_rule",
]


def _parse_brainvision_header(header_path: Path) -> tuple[int, np.dtype, np.ndarray]:
    source = header_path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=False,
        delimiters=("=",),
        comment_prefixes=(";",),
    )
    parser.optionxform = str
    first_section = source.find("[Common Infos]")
    if first_section < 0:
        raise ValueError(f"Missing [Common Infos] in {header_path}")
    parser.read_string(source[first_section:])
    common = parser["Common Infos"]
    if common.get("DataFormat", "").upper() != "BINARY":
        raise ValueError(f"Only binary BrainVision data are supported: {header_path}")
    if common.get("DataOrientation", "").upper() != "MULTIPLEXED":
        raise ValueError(f"Only multiplexed BrainVision data are supported: {header_path}")
    channel_count = int(common["NumberOfChannels"])
    binary_format = parser["Binary Infos"].get("BinaryFormat", "").upper()
    dtype_by_format = {
        "IEEE_FLOAT_32": np.dtype("<f4"),
        "INT_16": np.dtype("<i2"),
        "UINT_16": np.dtype("<u2"),
        "INT_32": np.dtype("<i4"),
    }
    if binary_format not in dtype_by_format:
        raise ValueError(f"Unsupported BrainVision BinaryFormat={binary_format!r}")
    resolutions = np.ones(channel_count, dtype=np.float64)
    unit_scales = np.ones(channel_count, dtype=np.float64)
    units = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6}
    channel_section = parser["Channel Infos"]
    for index in range(channel_count):
        value = channel_section.get(f"Ch{index + 1}")
        if value is None:
            raise ValueError(f"Missing Ch{index + 1} in {header_path}")
        fields = value.split(",")
        if len(fields) >= 3 and fields[2].strip():
            resolutions[index] = float(fields[2])
        if len(fields) >= 4 and fields[3].strip():
            unit = fields[3].strip()
            if unit not in units:
                raise ValueError(f"Unsupported BrainVision unit={unit!r}")
            unit_scales[index] = units[unit]
    return channel_count, dtype_by_format[binary_format], resolutions * unit_scales


class OfficialBrainVisionSegmentReader:
    """Read released arrays without modifying broken companion references.

    Several ChineseEEG1 headers contain a filename typo while the same-stem EEG
    binary is present. MNE correctly rejects such headers. For those files this
    reader parses the official header metadata and memory-maps the same-stem
    binary; neither the header nor EEG data is rewritten.
    """

    def __init__(self) -> None:
        self._raw_cache: dict[str, object] = {}
        self._memmap_cache: dict[str, tuple[np.memmap, np.ndarray]] = {}
        self.fallback_read_count = 0

    def __call__(self, eeg_file: str, start_sample: int, stop_sample: int) -> np.ndarray:
        binary_path = Path(eeg_file)
        header_path = binary_path.with_suffix(".vhdr")
        if not header_path.is_file():
            raise FileNotFoundError(f"BrainVision header not found: {header_path}")
        key = str(header_path.resolve())
        raw = self._raw_cache.get(key)
        if raw is None and key not in self._memmap_cache:
            try:
                import mne

                raw = mne.io.read_raw_brainvision(
                    header_path,
                    preload=False,
                    verbose="ERROR",
                )
                self._raw_cache[key] = raw
            except (FileNotFoundError, OSError):
                if not binary_path.is_file():
                    raise FileNotFoundError(
                        f"Header reference is broken and same-stem binary is absent: "
                        f"{binary_path}"
                    )
                channel_count, dtype, calibration = _parse_brainvision_header(
                    header_path
                )
                value_count = binary_path.stat().st_size // dtype.itemsize
                if value_count % channel_count:
                    raise ValueError(
                        f"BrainVision binary size is not divisible by {channel_count} "
                        f"channels: {binary_path}"
                    )
                sample_count = value_count // channel_count
                mapped = np.memmap(
                    binary_path,
                    dtype=dtype,
                    mode="r",
                    shape=(sample_count, channel_count),
                )
                self._memmap_cache[key] = (mapped, calibration)
                self.fallback_read_count += 1
        raw = self._raw_cache.get(key)
        if raw is not None:
            data = raw.get_data(start=start_sample, stop=stop_sample)
            return np.asarray(data, dtype=np.float32)
        mapped, calibration = self._memmap_cache[key]
        if start_sample < 0 or stop_sample > mapped.shape[0] or stop_sample <= start_sample:
            raise ValueError(
                f"Invalid EEG slice [{start_sample}, {stop_sample}) for {binary_path}"
            )
        data = np.asarray(mapped[start_sample:stop_sample], dtype=np.float64).T
        data *= calibration[:, None]
        return np.asarray(data, dtype=np.float32)


class SpanTextTargetProvider:
    """Load frozen text states and use raw offset mappings for span selection."""

    def __init__(
        self,
        *,
        feature_dir: str | Path,
        mode: TextTargetMode,
        maximum_cached_items: int = 256,
    ) -> None:
        if maximum_cached_items <= 0:
            raise ValueError("maximum_cached_items must be positive")
        self.feature_dir = Path(feature_dir)
        self.mode = mode
        self.maximum_cached_items = maximum_cached_items
        self._cache: OrderedDict[str, TextFeatureResult] = OrderedDict()

    def __call__(self, row: Mapping[str, object]) -> np.ndarray:
        is_local = self.mode.startswith("local_")
        feature_id = str(row["span_text_id"] if is_local else row["global_text_id"])
        result = self._load(feature_id)
        if is_local:
            raw_start = 0
            raw_stop = len(result.text)
            included = tuple(range(raw_stop))
        else:
            raw_start = int(row["span_start_char"])
            raw_stop = int(row["span_end_char"])
            sentence = str(row["source_sentence_text"])
            if result.text != sentence:
                raise ValueError(
                    f"Full-sentence feature text mismatch for {feature_id}"
                )
            included = tuple(
                index
                for index in chineseeeg1_clock_positions(sentence)
                if raw_start <= index < raw_stop
            )
        selection = select_text_span_by_offsets(
            result,
            raw_start=raw_start,
            raw_stop=raw_stop,
            included_character_indices=included,
            expected_character_count=int(row["span_char_count"]),
        )
        if self.mode.endswith("_character_sequence"):
            return np.asarray(selection.character_hidden_states, dtype=np.float32)
        pooling = "cls" if self.mode.endswith("_cls") else "mean"
        return np.asarray(
            pool_text_span(result, selection, pooling=pooling),
            dtype=np.float32,
        )

    def _load(self, feature_id: str) -> TextFeatureResult:
        cached = self._cache.pop(feature_id, None)
        if cached is not None:
            self._cache[feature_id] = cached
            return cached
        path = self.feature_dir / safe_artifact_filename(feature_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing text feature: {path}")
        result = load_text_features(path)
        if result.content_id != feature_id:
            raise ValueError(f"Text feature ID mismatch: {result.content_id} != {feature_id}")
        self._cache[feature_id] = result
        if len(self._cache) > self.maximum_cached_items:
            self._cache.popitem(last=False)
        return result


class ChineseEEG1SpanDataset(Dataset[dict[str, object]]):
    """One span length per dataset instance; model tensors never contain padding."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        partition: str,
        span_char_count: int,
        eeg_normalization: EEGNormalization = "per_span_channel_zscore",
        eeg_scaler: ChannelRobustScaler | None = None,
        text_target_provider: Callable[[Mapping[str, object]], np.ndarray] | None = None,
        eeg_reader: Callable[[str, int, int], np.ndarray] | None = None,
        strict_unseen_local_text: bool = False,
        semantic_only: bool = False,
        book_ids: Sequence[str] | None = None,
    ) -> None:
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown partition: {partition}")
        if span_char_count <= 0:
            raise ValueError("span_char_count must be positive")
        if eeg_normalization not in {
            "none",
            "per_span_channel_zscore",
            "train_recording_robust_clamp",
        }:
            raise ValueError(f"Unknown EEG normalization: {eeg_normalization}")
        if eeg_normalization == "train_recording_robust_clamp" and eeg_scaler is None:
            raise ValueError(
                "train_recording_robust_clamp requires a train-fitted scaler"
            )
        normalized_book_ids = (
            None
            if book_ids is None
            else tuple(dict.fromkeys(str(book_id) for book_id in book_ids))
        )
        if normalized_book_ids is not None and not normalized_book_ids:
            raise ValueError("book_ids cannot be empty when provided")
        index_path = Path(index_path)
        available_columns = set(pq.ParquetFile(index_path).schema.names)
        if semantic_only and "is_semantic_unit" not in available_columns:
            raise ValueError(
                "semantic_only=True requires is_semantic_unit in the span index"
            )
        filters: list[tuple[str, str, object]] = [
            ("split", "=", partition),
            ("span_char_count", "=", span_char_count),
        ]
        if semantic_only:
            filters.append(("is_semantic_unit", "=", True))
        if normalized_book_ids is not None:
            filters.append(("book_id", "in", list(normalized_book_ids)))
        table = pq.read_table(
            index_path,
            columns=DATASET_COLUMNS
            + [
                column
                for column in OPTIONAL_DATASET_COLUMNS
                if column in available_columns
            ],
            filters=filters,
        )
        if strict_unseen_local_text:
            if partition == "train":
                raise ValueError("strict_unseen_local_text is only a held-out sensitivity")
            train_filters: list[tuple[str, str, object]] = [
                ("split", "=", "train"),
                ("span_char_count", "=", span_char_count),
            ]
            if semantic_only:
                train_filters.append(("is_semantic_unit", "=", True))
            if normalized_book_ids is not None:
                train_filters.append(
                    ("book_id", "in", list(normalized_book_ids))
                )
            train_ids = pc.unique(
                pq.read_table(
                    index_path,
                    columns=["span_text_id"],
                    filters=train_filters,
                )["span_text_id"]
            )
            unseen_mask = pc.invert(pc.is_in(table["span_text_id"], value_set=train_ids))
            table = table.filter(unseen_mask)
        if not table.num_rows:
            raise ValueError(
                f"No spans for partition={partition}, length={span_char_count}"
            )
        if pc.any(pc.not_equal(table["padding_samples"], 0)).as_py():
            raise ValueError("Span index contains padding")
        if pc.any(table["exposes_padding_mask"]).as_py():
            raise ValueError("Span index exposes a padding mask")
        model_lengths = pc.unique(table["model_eeg_sample_count"]).to_pylist()
        if len(model_lengths) != 1:
            raise ValueError("A fixed-span dataset must have one model EEG length")

        subject_filters = None
        if normalized_book_ids is not None:
            subject_filters = [("book_id", "in", list(normalized_book_ids))]
        all_subjects = pc.unique(
            pq.read_table(
                index_path,
                columns=["subject_group_id"],
                filters=subject_filters,
            )["subject_group_id"]
        ).to_pylist()
        self.subject_index_by_group = {
            str(group_id): index for index, group_id in enumerate(sorted(all_subjects))
        }
        self.table = table.combine_chunks()
        self.partition = partition
        self.span_char_count = span_char_count
        self.semantic_only = semantic_only
        self.book_ids = normalized_book_ids
        self.model_eeg_sample_count = int(model_lengths[0])
        self.eeg_normalization = eeg_normalization
        self.eeg_scaler = eeg_scaler
        self.text_target_provider = text_target_provider
        self.eeg_reader = eeg_reader or OfficialBrainVisionSegmentReader()

    def __len__(self) -> int:
        return self.table.num_rows

    def __getitem__(self, index: int) -> dict[str, object]:
        row = {
            name: self.table[name][index].as_py()
            for name in self.table.column_names
        }
        eeg = self.eeg_reader(
            str(row["eeg_file"]),
            int(row["eeg_start_sample"]),
            int(row["eeg_stop_sample"]),
        )
        if eeg.ndim != 2 or eeg.shape[1] != int(row["source_eeg_sample_count"]):
            raise ValueError(
                f"EEG segment shape mismatch for {row['span_event_id']}: {eeg.shape}"
            )
        if self.eeg_normalization == "per_span_channel_zscore":
            eeg = _per_span_channel_zscore(eeg)
        elif self.eeg_normalization == "train_recording_robust_clamp":
            assert self.eeg_scaler is not None
            eeg = self.eeg_scaler.transform(str(row["eeg_file"]), eeg)
        eeg_tensor = torch.from_numpy(np.ascontiguousarray(eeg)).unsqueeze(0)
        eeg_tensor = F.interpolate(
            eeg_tensor,
            size=self.model_eeg_sample_count,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        output: dict[str, object] = {
            "record_id": str(row["record_id"]),
            "eeg": eeg_tensor,
            "span_event_id": str(row["span_event_id"]),
            "span_text_id": str(row["span_text_id"]),
            "global_text_id": str(row["global_text_id"]),
            "split_group_id": str(row["split_group_id"]),
            "subject_group_id": str(row["subject_group_id"]),
            "subject_index": self.subject_index_by_group[str(row["subject_group_id"])],
            "span_text": str(row["span_text"]),
            "span_char_count": self.span_char_count,
            "span_start_clock": int(row["span_start_clock"]),
            "span_end_clock": int(row["span_end_clock"]),
            "book_id": str(row["book_id"]),
            "stimulus_position": int(row["stimulus_position"]),
            "is_semantic_unit": bool(row.get("is_semantic_unit", False)),
            "semantic_unit_kind": str(row.get("semantic_unit_kind", "none")),
            "semantic_pos_pattern": row.get("semantic_pos_pattern"),
            "semantic_unit_rule": row.get("semantic_unit_rule"),
        }
        if self.text_target_provider is not None:
            target = self.text_target_provider(row)
            output["text"] = torch.from_numpy(
                np.array(target, dtype=np.float32, order="C", copy=True)
            )
        return output


def collate_fixed_character_spans(
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Collate only already-equal tensors; padding and masks are not an option."""

    if not items:
        raise ValueError("Cannot collate an empty batch")
    eeg_shapes = {tuple(item["eeg"].shape) for item in items}
    span_lengths = {int(item["span_char_count"]) for item in items}
    if len(eeg_shapes) != 1 or len(span_lengths) != 1:
        raise ValueError(
            f"Mixed fixed-span batch: eeg_shapes={eeg_shapes}, spans={span_lengths}"
        )
    batch: dict[str, object] = {
        "eeg": torch.stack([item["eeg"] for item in items]),
        "subject_index": torch.as_tensor(
            [int(item["subject_index"]) for item in items], dtype=torch.long
        ),
        "metadata": [
            {key: value for key, value in item.items() if key not in {"eeg", "text"}}
            for item in items
        ],
    }
    if all("text" in item for item in items):
        text_shapes = {tuple(item["text"].shape) for item in items}
        if len(text_shapes) != 1:
            raise ValueError(f"Text target shapes differ: {text_shapes}")
        batch["text"] = torch.stack([item["text"] for item in items])
    elif any("text" in item for item in items):
        raise ValueError("Only part of the batch contains text targets")
    if "padding_mask" in batch or "valid_samples" in batch:
        raise AssertionError("Fixed-span collate must never emit padding metadata")
    return batch


def _per_span_channel_zscore(eeg: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    mean = eeg.mean(axis=1, keepdims=True, dtype=np.float64)
    std = eeg.std(axis=1, keepdims=True, dtype=np.float64)
    return np.asarray((eeg - mean) / np.maximum(std, eps), dtype=np.float32)
