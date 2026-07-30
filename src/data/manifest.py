"""Typed manifest contract shared by data readers, training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

VALID_PARADIGMS = frozenset({"silent_reading", "passive_listening", "reading_aloud"})
VALID_SPLITS = frozenset({"train", "valid", "test"})


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One verified EEG block and its stimulus provenance.

    ``block_id`` identifies a concrete occurrence in a recording. ``content_id``
    identifies stimulus content independently of subject and paradigm.
    """

    record_id: str
    dataset_id: str
    paradigm: str
    book_id: str
    subject_id: str
    session_id: str
    run_id: str
    block_id: str
    content_id: str
    eeg_path: str
    eeg_start_sample: int
    eeg_stop_sample: int
    sampling_rate_hz: float
    text: str
    stimulus_position: int
    split: str | None = None
    material_variant: str | None = None
    audio_path: str | None = None
    audio_start_sec: float | None = None
    audio_stop_sec: float | None = None
    speaker_id: str | None = None
    alignment_method: str | None = None
    provenance: str | None = None

    @property
    def eeg_duration_sec(self) -> float:
        return (self.eeg_stop_sample - self.eeg_start_sample) / self.sampling_rate_hz

    @property
    def audio_duration_sec(self) -> float | None:
        if self.audio_start_sec is None or self.audio_stop_sec is None:
            return None
        return self.audio_stop_sec - self.audio_start_sec

    def validate(self) -> None:
        required_strings = {
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "paradigm": self.paradigm,
            "book_id": self.book_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "block_id": self.block_id,
            "content_id": self.content_id,
            "eeg_path": self.eeg_path,
        }
        empty = [name for name, value in required_strings.items() if not value]
        if empty:
            raise ValueError(f"Manifest record has empty required fields: {empty}")
        if self.paradigm not in VALID_PARADIGMS:
            raise ValueError(f"Unknown paradigm: {self.paradigm!r}")
        if self.split is not None and self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {self.split!r}")
        if self.eeg_start_sample < 0 or self.eeg_stop_sample <= self.eeg_start_sample:
            raise ValueError("EEG block must have positive, ordered sample boundaries")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.stimulus_position < 0:
            raise ValueError("stimulus_position must be non-negative")
        audio_bounds = (self.audio_start_sec, self.audio_stop_sec)
        if (audio_bounds[0] is None) != (audio_bounds[1] is None):
            raise ValueError("Audio start and stop must either both be present or both be absent")
        if audio_bounds[0] is not None:
            if audio_bounds[0] < 0 or audio_bounds[1] <= audio_bounds[0]:
                raise ValueError("Audio block must have positive, ordered time boundaries")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def validate_manifest_records(records: Iterable[ManifestRecord]) -> None:
    """Validate record uniqueness and content-group split integrity."""

    seen_records: set[str] = set()
    splits_by_content: dict[str, set[str]] = {}
    any_record = False

    for record in records:
        any_record = True
        record.validate()
        if record.record_id in seen_records:
            raise ValueError(f"Duplicate record_id: {record.record_id}")
        seen_records.add(record.record_id)
        if record.split is not None:
            splits_by_content.setdefault(record.content_id, set()).add(record.split)

    if not any_record:
        raise ValueError("Manifest is empty")

    leaked = {
        content_id: sorted(splits)
        for content_id, splits in splits_by_content.items()
        if len(splits) > 1
    }
    if leaked:
        preview = dict(list(leaked.items())[:5])
        raise ValueError(f"content_id appears in multiple splits: {preview}")

