"""Memory-mapped frozen character states for large short-text vocabularies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow.parquet as pq

CONSOLIDATED_TEXT_CACHE_SCHEMA = "consolidated-character-text-cache-v1"


class ConsolidatedSpanTextTargetProvider:
    """Return ``[characters, hidden]`` targets from length-specific memmaps."""

    def __init__(self, feature_dir: str | Path) -> None:
        self.feature_dir = Path(feature_dir)
        manifest_path = self.feature_dir / "extraction_manifest.json"
        index_path = self.feature_dir / "feature_index.parquet"
        if not manifest_path.is_file() or not index_path.is_file():
            raise FileNotFoundError("Consolidated text cache is incomplete")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("schema_version") != CONSOLIDATED_TEXT_CACHE_SCHEMA
            or not self.manifest.get("complete")
        ):
            raise ValueError("Consolidated text cache manifest is not complete")
        rows = pq.read_table(
            index_path,
            columns=["content_id", "text", "span_char_count", "row_index"],
        ).to_pylist()
        self._location_by_id: dict[str, tuple[str, int, int]] = {}
        for row in rows:
            content_id = str(row["content_id"])
            location = (
                str(row["text"]),
                int(row["span_char_count"]),
                int(row["row_index"]),
            )
            if content_id in self._location_by_id:
                raise ValueError(f"Duplicate consolidated feature ID: {content_id}")
            self._location_by_id[content_id] = location
        self._arrays: dict[int, np.ndarray] = {}

    def __call__(self, row: Mapping[str, object]) -> np.ndarray:
        content_id = str(row["span_text_id"])
        try:
            expected_text, length, index = self._location_by_id[content_id]
        except KeyError as error:
            raise FileNotFoundError(
                f"Missing consolidated text feature: {content_id}"
            ) from error
        if str(row["span_text"]) != expected_text:
            raise ValueError("Consolidated feature text does not match span metadata")
        if int(row["span_char_count"]) != length:
            raise ValueError("Consolidated feature length does not match span metadata")
        array = self._arrays.get(length)
        if array is None:
            path = self.feature_dir / f"character_hidden_k{length}.npy"
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3 or array.shape[1] != length:
                raise ValueError(f"Malformed consolidated array: {path}")
            self._arrays[length] = array
        return np.asarray(array[index], dtype=np.float32)


class ConsolidatedStaticSpanTextTargetProvider:
    """Mean-pool cached offset-aligned character states to one frozen ``[D]`` target."""

    def __init__(self, feature_dir: str | Path) -> None:
        self.sequence_provider = ConsolidatedSpanTextTargetProvider(feature_dir)

    @property
    def manifest(self) -> Mapping[str, object]:
        return self.sequence_provider.manifest

    def __call__(self, row: Mapping[str, object]) -> np.ndarray:
        sequence = self.sequence_provider(row)
        if sequence.ndim != 2 or not sequence.shape[0]:
            raise ValueError("Static text pooling requires [characters, hidden]")
        return np.asarray(sequence.mean(axis=0, dtype=np.float64), dtype=np.float32)
