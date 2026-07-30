"""Formal trial-manifest contract and Arrow schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

MANIFEST_SCHEMA_VERSION = "trial-manifest-v1"
SENTENCE_ID_VERSION = "sentence-v1"
GLOBAL_TEXT_ID_VERSION = "global-text-v1"
SPLIT_GROUP_ID_VERSION = "split-group-v1"
RECORD_ID_VERSION = "trial-record-v1"
BLOCK_ID_VERSION = "trial-block-v1"
EEG_END_SAMPLE_SEMANTICS = (
    "exclusive ROWE event-onset sample; EEG interval is [eeg_start_sample,eeg_end_sample)"
)

VALID_PARADIGMS = frozenset({"silent_reading", "passive_listening", "reading_aloud"})
VALID_SPLITS = frozenset({"train", "valid", "test"})
VALID_TEXT_ALIGNMENT_STATUSES = frozenset(
    {"exact", "normalized", "fuzzy", "manual", "unresolved"}
)


class QualityFlag(str, Enum):
    OK = "ok"
    UNRESOLVED_TEXT_ALIGNMENT = "unresolved_text_alignment"
    MISSING_TEXT = "missing_text"
    EVENT_TEXT_COUNT_MISMATCH = "event_text_count_mismatch"
    ORPHAN_ROW_EVENT_IN_RECORDING = "orphan_row_event_in_recording"
    MISSING_EVENTS_TSV = "missing_events_tsv"
    EVENTS_VMRK_COUNT_MISMATCH = "events_vmrk_count_mismatch"
    BROKEN_BRAINVISION_REFERENCE = "broken_brainvision_reference"
    INVALID_BRAINVISION_BINARY = "invalid_brainvision_binary"
    IMPLAUSIBLE_EEG_TRIAL_DURATION = "implausible_eeg_trial_duration"
    PRECHAPTER_TRIAL_UNRESOLVED = "prechapter_trial_unresolved"
    LOW_TEXT_ALIGNMENT_SCORE = "low_text_alignment_score"
    MATERIAL_VARIANT_UNCERTAIN = "material_variant_uncertain"
    RA_AUDIO_BOUNDARY_UNAVAILABLE = "ra_audio_boundary_unavailable"
    PL_AUDIO_MAPPING_UNVERIFIED = "pl_audio_mapping_unverified"
    AUDIO_EVENTS_MISSING = "audio_events_missing"
    AUDIO_FILE_COUNT_MISMATCH = "audio_file_count_mismatch"
    AUDIO_BOUNDARY_OUT_OF_RANGE = "audio_boundary_out_of_range"


def encode_quality_flags(flags: Iterable[QualityFlag | str]) -> str:
    values = {
        flag.value if isinstance(flag, QualityFlag) else str(flag)
        for flag in flags
        if str(flag)
    }
    values.discard(QualityFlag.OK.value)
    return "|".join(sorted(values)) if values else QualityFlag.OK.value


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    dataset_version: str
    paradigm: str
    subject_id: str
    session_id: str
    book_id: str
    chapter_id: int | None
    sentence_id: str
    global_text_id: str | None
    raw_text: str | None
    normalized_text: str | None
    text_hash: str | None
    raw_text_hash: str | None
    normalized_text_hash: str | None
    text_hash_basis: str
    char_count: int | None
    raw_char_count: int | None
    highlight_char_count: int | None
    char_count_method: str
    word_count: int | None
    word_count_method: str
    eeg_file: str
    eeg_start_sample: int
    eeg_end_sample: int
    eeg_end_sample_semantics: str
    eeg_duration_sec: float
    eeg_sampling_rate: float
    audio_file: str | None
    audio_start_sec: float | None
    audio_end_sec: float | None
    audio_duration_sec: float | None
    event_source: str
    alignment_source: str
    quality_flag: str
    split_group_id: str
    split: str
    record_id: str
    run_id: str
    block_id: str
    content_id: str | None
    material_variant: str | None
    speaker_id: str | None
    stimulus_position: int
    source_excel_file: str | None
    source_excel_row: int | None
    normalization_version: str
    normalization_trace: str
    matching_alias: str | None
    matching_alias_method: str
    text_alignment_status: str
    text_alignment_score: float | None
    global_text_alignment_status: str
    global_text_alignment_score: float | None
    audio_alignment_method: str
    audio_alignment_evidence: str
    event_pair_index: int
    preceding_chapter_event: str | None
    manifest_schema_version: str
    split_seed: int

    def validate(self) -> None:
        required = {
            "dataset_version": self.dataset_version,
            "paradigm": self.paradigm,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "book_id": self.book_id,
            "sentence_id": self.sentence_id,
            "eeg_file": self.eeg_file,
            "event_source": self.event_source,
            "alignment_source": self.alignment_source,
            "quality_flag": self.quality_flag,
            "split_group_id": self.split_group_id,
            "split": self.split,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "block_id": self.block_id,
        }
        empty = [name for name, value in required.items() if not value]
        if empty:
            raise ValueError(f"Manifest record has empty required fields: {empty}")
        if self.paradigm not in VALID_PARADIGMS:
            raise ValueError(f"Unknown paradigm: {self.paradigm!r}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {self.split!r}")
        if self.text_alignment_status not in VALID_TEXT_ALIGNMENT_STATUSES:
            raise ValueError(
                f"Unknown text alignment status: {self.text_alignment_status!r}"
            )
        if self.eeg_start_sample < 0 or self.eeg_end_sample <= self.eeg_start_sample:
            raise ValueError("EEG interval must be a positive half-open interval")
        if self.eeg_sampling_rate <= 0:
            raise ValueError("eeg_sampling_rate must be positive")
        expected_duration = (
            self.eeg_end_sample - self.eeg_start_sample
        ) / self.eeg_sampling_rate
        if abs(expected_duration - self.eeg_duration_sec) > 1e-9:
            raise ValueError("eeg_duration_sec disagrees with sample boundaries")
        if self.eeg_end_sample_semantics != EEG_END_SAMPLE_SEMANTICS:
            raise ValueError("Unexpected EEG end-sample semantics")
        audio_bounds = (self.audio_start_sec, self.audio_end_sec)
        if (audio_bounds[0] is None) != (audio_bounds[1] is None):
            raise ValueError("Audio start/end must both be present or both be null")
        if audio_bounds[0] is None:
            if self.audio_duration_sec is not None or self.audio_file is not None:
                raise ValueError("Null audio bounds require null file and duration")
        else:
            if audio_bounds[0] < 0 or audio_bounds[1] <= audio_bounds[0]:
                raise ValueError("Audio interval must be a positive half-open interval")
            if self.audio_duration_sec is None:
                raise ValueError("Audio duration is required with audio bounds")
        if self.raw_text is None:
            text_values = (
                self.normalized_text,
                self.text_hash,
                self.raw_text_hash,
                self.normalized_text_hash,
                self.char_count,
                self.raw_char_count,
                self.highlight_char_count,
                self.word_count,
            )
            if any(value is not None for value in text_values):
                raise ValueError("Missing raw_text requires all derived text fields to be null")
        if self.manifest_schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unexpected manifest schema version")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def validate_manifest_records(records: Iterable[ManifestRecord]) -> None:
    seen_records: set[str] = set()
    splits_by_group: dict[str, set[str]] = {}
    count = 0
    for record in records:
        count += 1
        record.validate()
        if record.record_id in seen_records:
            raise ValueError(f"Duplicate record_id: {record.record_id}")
        seen_records.add(record.record_id)
        splits_by_group.setdefault(record.split_group_id, set()).add(record.split)
    if count == 0:
        raise ValueError("Manifest is empty")
    leaked = {
        group_id: sorted(splits)
        for group_id, splits in splits_by_group.items()
        if len(splits) > 1
    }
    if leaked:
        raise ValueError(
            f"split_group_id appears in multiple splits: {dict(list(leaked.items())[:5])}"
        )


def manifest_arrow_schema():
    """Return the authoritative nullable PyArrow schema."""

    import pyarrow as pa

    string = pa.string()
    int32 = pa.int32()
    int64 = pa.int64()
    float64 = pa.float64()
    return pa.schema(
        [
            pa.field("dataset_version", string, nullable=False),
            pa.field("paradigm", string, nullable=False),
            pa.field("subject_id", string, nullable=False),
            pa.field("session_id", string, nullable=False),
            pa.field("book_id", string, nullable=False),
            pa.field("chapter_id", int32),
            pa.field("sentence_id", string, nullable=False),
            pa.field("global_text_id", string),
            pa.field("raw_text", string),
            pa.field("normalized_text", string),
            pa.field("text_hash", string),
            pa.field("raw_text_hash", string),
            pa.field("normalized_text_hash", string),
            pa.field("text_hash_basis", string, nullable=False),
            pa.field("char_count", int32),
            pa.field("raw_char_count", int32),
            pa.field("highlight_char_count", int32),
            pa.field("char_count_method", string, nullable=False),
            pa.field("word_count", int32),
            pa.field("word_count_method", string, nullable=False),
            pa.field("eeg_file", string, nullable=False),
            pa.field("eeg_start_sample", int64, nullable=False),
            pa.field("eeg_end_sample", int64, nullable=False),
            pa.field("eeg_end_sample_semantics", string, nullable=False),
            pa.field("eeg_duration_sec", float64, nullable=False),
            pa.field("eeg_sampling_rate", float64, nullable=False),
            pa.field("audio_file", string),
            pa.field("audio_start_sec", float64),
            pa.field("audio_end_sec", float64),
            pa.field("audio_duration_sec", float64),
            pa.field("event_source", string, nullable=False),
            pa.field("alignment_source", string, nullable=False),
            pa.field("quality_flag", string, nullable=False),
            pa.field("split_group_id", string, nullable=False),
            pa.field("split", string, nullable=False),
            pa.field("record_id", string, nullable=False),
            pa.field("run_id", string, nullable=False),
            pa.field("block_id", string, nullable=False),
            pa.field("content_id", string),
            pa.field("material_variant", string),
            pa.field("speaker_id", string),
            pa.field("stimulus_position", int32, nullable=False),
            pa.field("source_excel_file", string),
            pa.field("source_excel_row", int32),
            pa.field("normalization_version", string, nullable=False),
            pa.field("normalization_trace", string, nullable=False),
            pa.field("matching_alias", string),
            pa.field("matching_alias_method", string, nullable=False),
            pa.field("text_alignment_status", string, nullable=False),
            pa.field("text_alignment_score", float64),
            pa.field("global_text_alignment_status", string, nullable=False),
            pa.field("global_text_alignment_score", float64),
            pa.field("audio_alignment_method", string, nullable=False),
            pa.field("audio_alignment_evidence", string, nullable=False),
            pa.field("event_pair_index", int32, nullable=False),
            pa.field("preceding_chapter_event", string),
            pa.field("manifest_schema_version", string, nullable=False),
            pa.field("split_seed", int64, nullable=False),
        ],
        metadata={
            b"schema_version": MANIFEST_SCHEMA_VERSION.encode(),
            b"eeg_interval": b"half-open [start,end)",
            b"text_hash_basis": b"normalized_text sha256 UTF-8",
        },
    )
