"""Leakage-resistant fixed-character-span index for ChineseEEG1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data.chineseeeg1_timeline import (
    OFFICIAL_CHARACTER_DWELL_SEC,
    OFFICIAL_COMMIT,
    assert_timeline_method_allowed,
    chineseeeg1_clock_positions,
)
from data.protocol_splitting import make_subject_group_id
from data.semantic_units import (
    SEMANTIC_UNIT_RULE_VERSION,
    semantic_unit_annotations,
)

TimelineMethod = Literal["event_affine", "fixed_dwell_sensitivity", "sentence_weak"]
SPAN_SCHEMA_VERSION = "ce1-fixed-character-span-v1"
PARTITIONS = ("train", "validation", "test")


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join((prefix, *(str(part) for part in parts))).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class CharacterSpanSpec:
    span_lengths: tuple[int, ...] = (4, 6, 8)
    stride_characters: int = 1
    allowed_clock_starts: tuple[int, ...] | None = None
    timeline_method: TimelineMethod = "event_affine"
    neural_delay_ms: float = 200.0
    left_context_ms: float = 0.0
    right_context_ms: float = 0.0
    target_character_duration_ms: float = 350.0
    drop_if_context_crosses_row: bool = True
    include_low_confidence: bool = False
    annotate_semantic_units: bool = False

    def validate(self) -> None:
        if not self.span_lengths or any(length <= 0 for length in self.span_lengths):
            raise ValueError("span_lengths must contain positive integers")
        if len(set(self.span_lengths)) != len(self.span_lengths):
            raise ValueError("span_lengths must be unique")
        if self.stride_characters <= 0:
            raise ValueError("stride_characters must be positive")
        if self.allowed_clock_starts is not None:
            if len(set(self.allowed_clock_starts)) != len(
                self.allowed_clock_starts
            ):
                raise ValueError("allowed_clock_starts must be unique")
            if any(value < 0 for value in self.allowed_clock_starts):
                raise ValueError("allowed_clock_starts cannot contain negatives")
            if any(
                value % self.stride_characters
                for value in self.allowed_clock_starts
            ):
                raise ValueError(
                    "allowed_clock_starts must align to stride_characters"
                )
        if self.target_character_duration_ms <= 0:
            raise ValueError("target_character_duration_ms must be positive")
        if self.left_context_ms < 0 or self.right_context_ms < 0:
            raise ValueError("EEG context durations cannot be negative")
        if self.timeline_method not in {
            "event_affine",
            "fixed_dwell_sensitivity",
            "sentence_weak",
        }:
            raise ValueError(f"Unknown timeline method: {self.timeline_method}")
        if not self.drop_if_context_crosses_row:
            raise ValueError(
                "Cross-row context is forbidden because a row is the inherited split block"
            )


@dataclass(frozen=True, slots=True)
class ChineseEEG1CharacterSpan:
    span_schema_version: str
    span_event_id: str
    span_text_id: str
    record_id: str
    block_id: str
    content_id: str
    split_group_id: str
    split: str
    global_text_id: str
    subject_group_id: str
    subject_id: str
    session_id: str
    book_id: str
    chapter_id: str | None
    run_id: str
    sentence_id: str
    stimulus_position: int
    source_unit_kind: str
    source_sentence_text: str
    preceding_context_text: str | None
    following_context_text: str | None
    span_text: str
    span_surface_text: str
    span_char_count: int
    span_start_char: int
    span_end_char: int
    span_start_clock: int
    span_end_clock: int
    span_position_fraction: float
    eeg_file: str
    eeg_sampling_rate_hz: int
    source_row_start_sample: int
    source_row_stop_sample: int
    eeg_start_sample: int
    eeg_stop_sample: int
    source_eeg_sample_count: int
    model_eeg_sample_count: int
    neural_delay_ms: float
    left_context_ms: float
    right_context_ms: float
    timeline_method: str
    timeline_source: str
    timeline_rule: str
    effective_character_interval_sec: float
    configured_clock_boundary_disagreement_sec: float
    alignment_confidence: str
    exact_character_onsets_observed: bool
    resampling_method: str
    padding_samples: int
    exposes_padding_mask: bool
    brainvision_reference_repair_required: bool
    source_quality_flag: str
    is_semantic_unit: bool = False
    semantic_unit_kind: str = "none"
    semantic_pos_pattern: str | None = None
    semantic_unit_rule: str | None = None

    def validate(self) -> None:
        if self.span_schema_version != SPAN_SCHEMA_VERSION:
            raise ValueError("Unexpected span schema version")
        if self.split not in PARTITIONS:
            raise ValueError(f"Invalid partition: {self.split}")
        if self.span_end_char <= self.span_start_char:
            raise ValueError("Raw character offsets must be ordered")
        if self.span_end_clock - self.span_start_clock != self.span_char_count:
            raise ValueError("Clock span and span_char_count disagree")
        if len(self.span_text) != self.span_char_count:
            raise ValueError("span_text must contain exactly the clock characters")
        if self.eeg_stop_sample <= self.eeg_start_sample:
            raise ValueError("EEG boundaries must be ordered")
        if self.source_eeg_sample_count != self.eeg_stop_sample - self.eeg_start_sample:
            raise ValueError("source_eeg_sample_count is inconsistent")
        if self.model_eeg_sample_count <= 0:
            raise ValueError("model_eeg_sample_count must be positive")
        if self.padding_samples != 0 or self.exposes_padding_mask:
            raise ValueError("Fixed-span samples may not expose padding")
        if not (
            self.source_row_start_sample <= self.eeg_start_sample
            and self.eeg_stop_sample <= self.source_row_stop_sample
        ):
            raise ValueError("EEG span crosses its source row block")


def load_protocol_record_partitions(path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("setting") not in {
        "A_text_unseen_subject_seen",
        "A_unseen_text_subjects_visible",
        "seen_text_record_occurrence_split",
    }:
        raise ValueError(
            "ChineseEEG1 primary spans require Setting A unseen-text partitions"
        )
    assignments: dict[str, str] = {}
    for partition in PARTITIONS:
        try:
            record_ids = artifact["partitions"][partition]["record_ids"]
        except KeyError as error:
            raise ValueError(f"Malformed split artifact: missing {error.args[0]}") from error
        for record_id in record_ids:
            if record_id in assignments:
                raise ValueError(f"Duplicate record assignment: {record_id}")
            assignments[str(record_id)] = partition
    return assignments, artifact


def _context_by_record(rows: Sequence[Mapping[str, object]]) -> dict[str, tuple[str | None, str | None]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset_version") == "ChineseEEG1" and row.get("paradigm") == "silent_reading":
            groups[
                (str(row["subject_id"]), str(row["session_id"]), str(row["run_id"]))
            ].append(row)
    context: dict[str, tuple[str | None, str | None]] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["stimulus_position"]))
        for index, row in enumerate(ordered):
            previous = str(ordered[index - 1]["raw_text"]) if index else None
            following = str(ordered[index + 1]["raw_text"]) if index + 1 < len(ordered) else None
            context[str(row["record_id"])] = (previous, following)
    return context


def _row_is_eligible(row: Mapping[str, object]) -> tuple[bool, str | None]:
    if row.get("dataset_version") != "ChineseEEG1" or row.get("paradigm") != "silent_reading":
        return False, "not_chineseeeg1_silent_reading"
    if not row.get("raw_text"):
        return False, "missing_text"
    if row.get("text_alignment_status") != "exact":
        return False, "non_exact_stimulus_row_alignment"
    quality = str(row.get("quality_flag") or "")
    if "orphan_row_event_in_recording" in quality:
        return False, "orphan_row_event"
    if row.get("eeg_start_sample") is None or row.get("eeg_end_sample") is None:
        return False, "missing_eeg_boundary"
    if float(row.get("eeg_sampling_rate") or 0.0) <= 0:
        return False, "invalid_sampling_rate"
    return True, None


def _alignment_confidence(
    effective_interval_sec: float,
    *,
    configured_interval_sec: float = OFFICIAL_CHARACTER_DWELL_SEC,
) -> str:
    difference = abs(effective_interval_sec - configured_interval_sec)
    if difference <= 0.10:
        return "medium"
    return "low"


def _clock_boundary(
    *,
    method: TimelineMethod,
    row_start: int,
    row_stop: int,
    clock_index: int,
    clock_count: int,
    sampling_rate: int,
) -> int:
    if method == "event_affine":
        return row_start + round((row_stop - row_start) * clock_index / clock_count)
    if method == "fixed_dwell_sensitivity":
        return row_start + round(clock_index * OFFICIAL_CHARACTER_DWELL_SEC * sampling_rate)
    raise ValueError("sentence_weak does not define character boundaries")


def iter_chineseeeg1_character_spans(
    rows: Sequence[Mapping[str, object]],
    *,
    record_partitions: Mapping[str, str],
    timeline_audit: Mapping[str, Any],
    spec: CharacterSpanSpec = CharacterSpanSpec(),
    counters: Counter[str] | None = None,
) -> Iterator[ChineseEEG1CharacterSpan]:
    """Generate spans after records have inherited a content-level partition."""

    spec.validate()
    assert_timeline_method_allowed(timeline_audit, spec.timeline_method)
    if spec.timeline_method == "sentence_weak":
        raise ValueError(
            "sentence_weak is a fallback for a separate MIL/monotonic dataset; "
            "it cannot emit claimed character spans"
        )
    counts = counters if counters is not None else Counter()
    contexts = _context_by_record(rows)
    for row in rows:
        eligible, reason = _row_is_eligible(row)
        if not eligible:
            counts[f"excluded_record:{reason}"] += 1
            continue
        record_id = str(row["record_id"])
        split = record_partitions.get(record_id)
        if split is None:
            counts["excluded_record:not_selected_by_protocol"] += 1
            continue
        if split not in PARTITIONS:
            raise ValueError(f"Unknown split assignment for {record_id}: {split}")
        text = str(row["raw_text"])
        positions = chineseeeg1_clock_positions(text)
        if not positions:
            counts["excluded_record:no_clock_character"] += 1
            continue
        row_start = int(row["eeg_start_sample"])
        row_stop = int(row["eeg_end_sample"])
        sampling_rate = int(round(float(row["eeg_sampling_rate"])))
        row_duration_sec = (row_stop - row_start) / sampling_rate
        effective_interval = row_duration_sec / len(positions)
        confidence = _alignment_confidence(effective_interval)
        if confidence == "low" and not spec.include_low_confidence:
            counts["excluded_record:low_alignment_confidence"] += 1
            continue
        previous, following = contexts[record_id]
        semantic_by_raw_interval = (
            semantic_unit_annotations(text)
            if spec.annotate_semantic_units
            else {}
        )
        delay_samples = round(spec.neural_delay_ms / 1000.0 * sampling_rate)
        left_samples = round(spec.left_context_ms / 1000.0 * sampling_rate)
        right_samples = round(spec.right_context_ms / 1000.0 * sampling_rate)
        for span_length in spec.span_lengths:
            if len(positions) < span_length:
                counts[f"no_span:{span_length}"] += 1
                continue
            model_samples = round(
                (
                    span_length * spec.target_character_duration_ms
                    + spec.left_context_ms
                    + spec.right_context_ms
                )
                / 1000.0
                * sampling_rate
            )
            for clock_start in range(
                0,
                len(positions) - span_length + 1,
                spec.stride_characters,
            ):
                if (
                    spec.allowed_clock_starts is not None
                    and clock_start not in spec.allowed_clock_starts
                ):
                    counts[f"excluded_span:{span_length}:clock_start_not_allowed"] += 1
                    continue
                clock_stop = clock_start + span_length
                visual_start = _clock_boundary(
                    method=spec.timeline_method,
                    row_start=row_start,
                    row_stop=row_stop,
                    clock_index=clock_start,
                    clock_count=len(positions),
                    sampling_rate=sampling_rate,
                )
                visual_stop = _clock_boundary(
                    method=spec.timeline_method,
                    row_start=row_start,
                    row_stop=row_stop,
                    clock_index=clock_stop,
                    clock_count=len(positions),
                    sampling_rate=sampling_rate,
                )
                eeg_start = visual_start + delay_samples - left_samples
                if spec.timeline_method == "fixed_dwell_sensitivity":
                    # A fixed-duration sensitivity must be fixed in the released
                    # source samples, not only after model-input resampling.  At
                    # 256 Hz, 2.2 seconds is represented by the nearest integer
                    # count (563 samples); independently rounding both endpoints
                    # would otherwise produce 564 samples for the k=4 schedule.
                    eeg_stop = eeg_start + model_samples
                else:
                    eeg_stop = visual_stop + delay_samples + right_samples
                if eeg_start < row_start or eeg_stop > row_stop:
                    counts[f"excluded_span:{span_length}:crosses_row_after_delay_context"] += 1
                    continue
                raw_start = positions[clock_start]
                raw_stop = positions[clock_stop - 1] + 1
                clock_characters = tuple(
                    text[index] for index in positions[clock_start:clock_stop]
                )
                span_text = "".join(clock_characters)
                semantic = semantic_by_raw_interval.get((raw_start, raw_stop))
                span_text_id = _stable_id("ce1-span-text-v1", span_text)
                span_event_id = _stable_id(
                    "ce1-span-event-v1",
                    row["global_text_id"],
                    clock_start,
                    clock_stop,
                )
                fixed_start_sec = clock_start * OFFICIAL_CHARACTER_DWELL_SEC
                fixed_stop_sec = clock_stop * OFFICIAL_CHARACTER_DWELL_SEC
                affine_start_sec = clock_start * effective_interval
                affine_stop_sec = clock_stop * effective_interval
                boundary_disagreement = max(
                    abs(affine_start_sec - fixed_start_sec),
                    abs(affine_stop_sec - fixed_stop_sec),
                )
                quality = str(row.get("quality_flag") or "ok")
                item = ChineseEEG1CharacterSpan(
                    span_schema_version=SPAN_SCHEMA_VERSION,
                    span_event_id=span_event_id,
                    span_text_id=span_text_id,
                    record_id=record_id,
                    block_id=str(row["block_id"]),
                    content_id=str(row["content_id"]),
                    split_group_id=str(row["split_group_id"]),
                    split=split,
                    global_text_id=str(row["global_text_id"]),
                    subject_group_id=make_subject_group_id(row),
                    subject_id=str(row["subject_id"]),
                    session_id=str(row["session_id"]),
                    book_id=str(row["book_id"]),
                    chapter_id=(
                        None if row.get("chapter_id") is None else str(row["chapter_id"])
                    ),
                    run_id=str(row["run_id"]),
                    sentence_id=str(row["sentence_id"]),
                    stimulus_position=int(row["stimulus_position"]),
                    source_unit_kind="stimulus_display_row_not_linguistic_word",
                    source_sentence_text=text,
                    preceding_context_text=previous,
                    following_context_text=following,
                    span_text=span_text,
                    span_surface_text=text[raw_start:raw_stop],
                    span_char_count=span_length,
                    span_start_char=raw_start,
                    span_end_char=raw_stop,
                    span_start_clock=clock_start,
                    span_end_clock=clock_stop,
                    span_position_fraction=(
                        (clock_start + 0.5 * span_length) / len(positions)
                    ),
                    eeg_file=str(row["eeg_file"]),
                    eeg_sampling_rate_hz=sampling_rate,
                    source_row_start_sample=row_start,
                    source_row_stop_sample=row_stop,
                    eeg_start_sample=eeg_start,
                    eeg_stop_sample=eeg_stop,
                    source_eeg_sample_count=eeg_stop - eeg_start,
                    model_eeg_sample_count=model_samples,
                    neural_delay_ms=spec.neural_delay_ms,
                    left_context_ms=spec.left_context_ms,
                    right_context_ms=spec.right_context_ms,
                    timeline_method=spec.timeline_method,
                    timeline_source=(
                        "official_presentation_code@"
                        f"{OFFICIAL_COMMIT}+ROWS_ROWE_event_pair"
                    ),
                    timeline_rule=(
                        "affine subdivision of observed ROWS-ROWE by exact official "
                        "presentation-clock character count"
                        if spec.timeline_method == "event_affine"
                        else "ROWS + clock_index * configured 0.35 s"
                    ),
                    effective_character_interval_sec=effective_interval,
                    configured_clock_boundary_disagreement_sec=boundary_disagreement,
                    alignment_confidence=confidence,
                    exact_character_onsets_observed=False,
                    resampling_method="linear_interpolation_to_fixed_span_length",
                    padding_samples=0,
                    exposes_padding_mask=False,
                    brainvision_reference_repair_required=(
                        "broken_brainvision_reference" in quality
                    ),
                    source_quality_flag=quality,
                    is_semantic_unit=semantic is not None,
                    semantic_unit_kind=(semantic.kind if semantic is not None else "none"),
                    semantic_pos_pattern=(
                        semantic.pos_pattern if semantic is not None else None
                    ),
                    semantic_unit_rule=(
                        SEMANTIC_UNIT_RULE_VERSION
                        if spec.annotate_semantic_units
                        else None
                    ),
                )
                item.validate()
                counts[f"span:{span_length}:{split}"] += 1
                yield item


def write_character_span_parquet(
    path: str | Path,
    spans: Iterable[ChineseEEG1CharacterSpan],
    *,
    batch_size: int = 20_000,
) -> tuple[int, str]:
    """Stream spans to Parquet without materializing the full corpus."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    batch: list[dict[str, object]] = []
    count = 0
    try:
        for span in spans:
            batch.append(asdict(span))
            if len(batch) < batch_size:
                continue
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            count += len(batch)
            batch.clear()
        if batch:
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
            count += len(batch)
        if writer is None:
            raise ValueError("No character spans were generated")
    finally:
        if writer is not None:
            writer.close()
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return count, digest


def read_character_spans(
    path: str | Path,
    *,
    split: str | None = None,
    span_char_count: int | None = None,
) -> list[ChineseEEG1CharacterSpan]:
    filters: list[tuple[str, str, object]] = []
    if split is not None:
        if split not in PARTITIONS:
            raise ValueError(f"Unknown split: {split}")
        filters.append(("split", "=", split))
    if span_char_count is not None:
        filters.append(("span_char_count", "=", span_char_count))
    table = pq.read_table(path, filters=filters or None)
    output = [ChineseEEG1CharacterSpan(**row) for row in table.to_pylist()]
    for span in output:
        span.validate()
    return output


def audit_span_index(
    path: str | Path,
    *,
    counters: Mapping[str, int],
    spec: CharacterSpanSpec,
    timeline_audit_path: str | Path,
    split_artifact_path: str | Path,
) -> dict[str, Any]:
    import pyarrow.compute as pc

    columns = [
        "record_id",
        "split_group_id",
        "span_event_id",
        "span_text_id",
        "split",
        "span_char_count",
        "span_start_clock",
        "source_eeg_sample_count",
        "model_eeg_sample_count",
        "padding_samples",
        "exposes_padding_mask",
        "exact_character_onsets_observed",
    ]
    table = pq.read_table(path, columns=columns)
    partition_counts = {
        str(item["values"]): int(item["counts"])
        for item in pc.value_counts(table["split"]).to_pylist()
    }
    span_length_counts = {
        int(item["values"]): int(item["counts"])
        for item in pc.value_counts(table["span_char_count"]).to_pylist()
    }
    group_partitions = table.group_by("split_group_id").aggregate(
        [("split", "count_distinct")]
    )
    event_partitions = table.group_by("span_event_id").aggregate(
        [("split", "count_distinct")]
    )
    sample_groups = table.group_by("span_char_count").aggregate(
        [
            ("source_eeg_sample_count", "count_distinct"),
            ("model_eeg_sample_count", "count_distinct"),
        ]
    )
    if any(
        value != 1
        for value in sample_groups["model_eeg_sample_count_count_distinct"].to_pylist()
    ):
        samples_by_length: dict[int, list[int]] = {}
        for length in sorted(span_length_counts):
            mask = pc.equal(table["span_char_count"], length)
            samples_by_length[length] = sorted(
                int(value)
                for value in pc.unique(
                    pc.filter(table["model_eeg_sample_count"], mask)
                ).to_pylist()
            )
    else:
        samples_by_length = {}
        for length in sorted(span_length_counts):
            mask = pc.equal(table["span_char_count"], length)
            samples_by_length[length] = [
                int(
                    pc.min(pc.filter(table["model_eeg_sample_count"], mask)).as_py()
                )
            ]
    source_samples_by_length: dict[int, list[int]] = {}
    for length in sorted(span_length_counts):
        mask = pc.equal(table["span_char_count"], length)
        source_samples_by_length[length] = sorted(
            int(value)
            for value in pc.unique(
                pc.filter(table["source_eeg_sample_count"], mask)
            ).to_pylist()
        )
    record_counts = table.group_by("record_id").aggregate(
        [("span_event_id", "count")]
    )["span_event_id_count"].to_pylist()
    unique_local_text_by_partition = {}
    for partition in PARTITIONS:
        mask = pc.equal(table["split"], partition)
        unique_local_text_by_partition[partition] = {
            str(value)
            for value in pc.unique(pc.filter(table["span_text_id"], mask)).to_pylist()
        }
    train_local = unique_local_text_by_partition["train"]
    return {
        "span_schema_version": SPAN_SCHEMA_VERSION,
        "index_path": Path(path).as_posix(),
        "index_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "timeline_audit_path": Path(timeline_audit_path).as_posix(),
        "split_artifact_path": Path(split_artifact_path).as_posix(),
        "spec": asdict(spec),
        "counts": {
            "span_count": table.num_rows,
            "by_partition": dict(sorted(partition_counts.items())),
            "by_span_length": {
                str(key): value for key, value in sorted(span_length_counts.items())
            },
            "unique_span_event_count": int(pc.count_distinct(table["span_event_id"]).as_py()),
            "unique_local_text_count": int(pc.count_distinct(table["span_text_id"]).as_py()),
            "generation_counters": dict(sorted(counters.items())),
        },
        "leakage_checks": {
            "split_group_cross_partition_count": sum(
                value > 1
                for value in group_partitions["split_count_distinct"].to_pylist()
            ),
            "span_event_cross_partition_count": sum(
                value > 1
                for value in event_partitions["split_count_distinct"].to_pylist()
            ),
            "model_sample_counts_per_span_length": {
                str(key): values for key, values in sorted(samples_by_length.items())
            },
            "source_sample_counts_per_span_length": {
                str(key): values
                for key, values in sorted(source_samples_by_length.items())
            },
            "all_source_and_model_sample_counts_equal": bool(
                pc.all(
                    pc.equal(
                        table["source_eeg_sample_count"],
                        table["model_eeg_sample_count"],
                    )
                ).as_py()
            ),
            "all_clock_starts_stride_aligned": all(
                int(value) % spec.stride_characters == 0
                for value in table["span_start_clock"].to_pylist()
            ),
            "maximum_spans_per_source_record": max(
                (int(value) for value in record_counts), default=0
            ),
            "all_padding_samples_zero": bool(
                pc.all(pc.equal(table["padding_samples"], 0)).as_py()
            ),
            "any_padding_mask_exposed": bool(
                pc.any(table["exposes_padding_mask"]).as_py()
            ),
            "all_exact_character_onsets_false": bool(
                pc.all(pc.invert(table["exact_character_onsets_observed"])).as_py()
            ),
        },
        "local_text_overlap": {
            "validation_local_text_seen_in_train_count": len(
                unique_local_text_by_partition["validation"] & train_local
            ),
            "test_local_text_seen_in_train_count": len(
                unique_local_text_by_partition["test"] & train_local
            ),
            "note": (
                "The primary protocol guarantees unseen parent text blocks. Short "
                "4/6/8-character strings may recur across different parent blocks; "
                "report a strict-unseen-local-text sensitivity analysis as well."
            ),
        },
    }


def render_span_audit_markdown(audit: Mapping[str, Any]) -> str:
    counts = audit["counts"]
    leakage = audit["leakage_checks"]
    overlap = audit["local_text_overlap"]
    return f"""# ChineseEEG1 fixed-character-span index audit

- Schema: `{audit['span_schema_version']}`
- Spans: {counts['span_count']:,}
- Unique span events: {counts['unique_span_event_count']:,}
- Unique local texts: {counts['unique_local_text_count']:,}
- Split groups crossing partitions: {leakage['split_group_cross_partition_count']}
- Span events crossing partitions: {leakage['span_event_cross_partition_count']}
- All padding counts zero: `{leakage['all_padding_samples_zero']}`
- Any padding mask exposed: `{leakage['any_padding_mask_exposed']}`

## Partition counts

| partition | spans |
|---|---:|
""" + "\n".join(
        f"| {partition} | {count:,} |"
        for partition, count in counts["by_partition"].items()
    ) + f"""

## Fixed model length

`model_sample_counts_per_span_length = {json.dumps(leakage['model_sample_counts_per_span_length'], sort_keys=True)}`

`source_sample_counts_per_span_length = {json.dumps(leakage['source_sample_counts_per_span_length'], sort_keys=True)}`

- Source/model sample counts identical: `{leakage['all_source_and_model_sample_counts_equal']}`
- All starts aligned to configured stride: `{leakage['all_clock_starts_stride_aligned']}`
- Maximum spans per source record: {leakage['maximum_spans_per_source_record']}

EEG segments are linearly resampled to the one fixed model length for their span
size. No padding or attention mask is returned to the model.

## Local-text overlap sensitivity

- Validation local strings also seen in train: {overlap['validation_local_text_seen_in_train_count']:,}
- Test local strings also seen in train: {overlap['test_local_text_seen_in_train_count']:,}

The Setting-A split guarantees unseen parent content blocks, not that every short
4/6/8-character string is unique. Main results must state this distinction and
also report a strict-unseen-local-string sensitivity subset.
"""
