"""Protocol-bound physical-time windows for ChineseEEG2 passive listening."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq

from data.protocol_splitting import artifact_sha256, make_subject_group_id

PL_WINDOW_SCHEMA_VERSION = "pl-speech-window-v1"
PL_REQUIRED_COLUMNS = (
    "record_id",
    "block_id",
    "split_group_id",
    "dataset_version",
    "paradigm",
    "subject_id",
    "session_id",
    "run_id",
    "speaker_id",
    "eeg_file",
    "eeg_sampling_rate",
    "eeg_start_sample",
    "eeg_end_sample",
    "audio_file",
    "audio_start_sec",
    "audio_end_sec",
    "audio_alignment_method",
    "audio_alignment_evidence",
    "quality_flag",
    "stimulus_position",
    "char_count",
    "eeg_duration_sec",
)


@dataclass(frozen=True, slots=True)
class PLSpeechWindowSpec:
    window_sec: float = 3.0
    stride_sec: float = 3.0
    delay_ms: float = 0.0
    tail_policy: str = "drop"

    def validate(self) -> None:
        if self.window_sec <= 0 or self.stride_sec <= 0:
            raise ValueError("window_sec and stride_sec must be positive")
        if self.tail_policy != "drop":
            raise ValueError(
                "PL speech baseline requires drop-tail windows; padding is forbidden"
            )


@dataclass(frozen=True, slots=True)
class PLSpeechWindow:
    window_schema_version: str
    window_id: str
    record_id: str
    block_id: str
    split_group_id: str
    split: str
    subject_group_id: str
    subject_id: str
    session_id: str
    run_id: str
    speaker_id: str
    stimulus_position: int
    char_count: int | None
    eeg_file: str
    eeg_sampling_rate_hz: int
    eeg_start_sample: int
    eeg_stop_sample: int
    eeg_sample_count: int
    valid_eeg_samples: int
    padded_eeg_samples: int
    audio_file: str
    audio_source_sample_rate_hz: int
    audio_start_sample: int
    audio_stop_sample: int
    audio_start_sec: float
    audio_stop_sec: float
    audio_target_id: str
    window_offset_sec: float
    window_sec: float
    stride_sec: float
    eeg_delay_ms: float
    source_trial_eeg_duration_sec: float
    overlap_source: str
    quality_flag: str


def load_pl_manifest_rows(path: str | Path) -> list[dict[str, object]]:
    return pq.read_table(
        path,
        columns=list(PL_REQUIRED_COLUMNS),
        filters=[("paradigm", "=", "passive_listening")],
    ).to_pylist()


def load_record_partitions(path: str | Path) -> tuple[dict[str, str], dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not str(payload.get("setting", "")).startswith("A_"):
        raise ValueError("PL baseline currently requires Setting A unseen-text split")
    assignments: dict[str, str] = {}
    for partition in ("train", "validation", "test"):
        for record_id in payload["partitions"][partition]["record_ids"]:
            previous = assignments.setdefault(str(record_id), partition)
            if previous != partition:
                raise ValueError(f"record_id crosses protocol partitions: {record_id}")
    if len(assignments) != int(payload["manifest_row_count"]):
        raise ValueError("Setting A split does not account for every manifest record")
    return assignments, payload


def build_pl_speech_windows(
    rows: Sequence[Mapping[str, object]],
    *,
    record_partitions: Mapping[str, str],
    spec: PLSpeechWindowSpec,
    manifest_path: str,
    split_artifact_path: str,
    audio_info_provider: Callable[[str], tuple[int, int]] | None = None,
) -> tuple[list[PLSpeechWindow], dict[str, object]]:
    """Create synchronized EEG/audio windows entirely inside one trial block."""

    spec.validate()
    if not rows:
        raise ValueError("PL manifest rows cannot be empty")
    info_provider = audio_info_provider or _soundfile_info
    info_cache: dict[str, tuple[int, int]] = {}
    windows: list[PLSpeechWindow] = []
    excluded: dict[str, list[str]] = defaultdict(list)
    records_with_windows: set[str] = set()

    for row in rows:
        record_id = str(row["record_id"])
        if record_id not in record_partitions:
            raise ValueError(f"PL record missing from split artifact: {record_id}")
        if row["dataset_version"] != "ChineseEEG2" or row["paradigm"] != (
            "passive_listening"
        ):
            raise ValueError("PL window builder received an out-of-scope row")
        audio_file = row["audio_file"]
        if (
            audio_file is None
            or row["audio_start_sec"] is None
            or row["audio_end_sec"] is None
        ):
            excluded["unverified_or_missing_audio_alignment"].append(record_id)
            continue
        audio_path = str(audio_file)
        if audio_path not in info_cache:
            try:
                info_cache[audio_path] = info_provider(audio_path)
            except (FileNotFoundError, OSError, RuntimeError):
                excluded["audio_file_unreadable"].append(record_id)
                continue
        audio_rate, audio_frame_count = info_cache[audio_path]
        if audio_rate <= 0 or audio_frame_count <= 0:
            raise ValueError(f"Invalid audio metadata for {audio_path}")
        audio_block_stop = round(float(row["audio_end_sec"]) * audio_rate)
        if audio_block_stop > audio_frame_count:
            excluded["audio_bounds_exceed_file"].append(record_id)
            continue

        row_windows = _windows_for_row(
            row,
            split=record_partitions[record_id],
            spec=spec,
            audio_sample_rate_hz=audio_rate,
            audio_frame_count=audio_frame_count,
        )
        if not row_windows:
            excluded["shorter_than_window_after_delay"].append(record_id)
            continue
        windows.extend(row_windows)
        records_with_windows.add(record_id)

    windows.sort(key=lambda window: window.window_id)
    _validate_window_integrity(windows)
    excluded_ids = [
        record_id
        for reason in sorted(excluded)
        for record_id in sorted(excluded[reason])
    ]
    if len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError("A PL record has more than one exclusion reason")
    if len(records_with_windows) + len(excluded_ids) != len(rows):
        raise ValueError("PL record-level selected + excluded accounting is incomplete")

    partition_counts = Counter(window.split for window in windows)
    group_partitions: dict[str, set[str]] = defaultdict(set)
    target_partitions: dict[str, set[str]] = defaultdict(set)
    for window in windows:
        group_partitions[window.split_group_id].add(window.split)
        target_partitions[window.audio_target_id].add(window.split)
    audit = {
        "window_schema_version": PL_WINDOW_SCHEMA_VERSION,
        "manifest_path": Path(manifest_path).as_posix(),
        "split_artifact_path": Path(split_artifact_path).as_posix(),
        "split_artifact_sha256": (
            artifact_sha256(split_artifact_path)
            if Path(split_artifact_path).is_file()
            else None
        ),
        "paradigm": "passive_listening",
        "window_spec": asdict(spec),
        "physical_time_mapping": {
            "audio_relative_start_sec": "window_offset_sec",
            "eeg_relative_start_sec": "window_offset_sec + eeg_delay_ms / 1000",
            "interpretation": (
                "positive delay pairs an audio span with a later EEG span; both "
                "spans remain within the same manifest trial block"
            ),
            "sample_quantization": "round(seconds * sampling_rate)",
        },
        "eligibility_policy": {
            "requires_verified_audio_bounds": True,
            "requires_readable_audio_file": True,
            "tail_policy": "drop",
            "quality_filtering_other_than_audio_alignment": False,
        },
        "shortcut_relevant_fields": {
            "window_duration_sec_unique": sorted(
                {window.window_sec for window in windows}
            ),
            "eeg_sample_count_unique": sorted(
                {window.eeg_sample_count for window in windows}
            ),
            "padded_eeg_samples_unique": sorted(
                {window.padded_eeg_samples for window in windows}
            ),
            "stimulus_position_preserved_for_control_only": True,
            "subject_group_id_preserved_for_control_only": True,
            "encoder_receives_shortcut_metadata": False,
        },
        "counts": {
            "input_pl_record_count": len(rows),
            "records_with_windows": len(records_with_windows),
            "excluded_record_count": len(excluded_ids),
            "window_count": len(windows),
            "audio_target_count": len(
                {window.audio_target_id for window in windows}
            ),
            "content_group_count": len(
                {window.split_group_id for window in windows}
            ),
            "subject_group_count": len(
                {window.subject_group_id for window in windows}
            ),
            "window_counts_by_partition": {
                partition: partition_counts.get(partition, 0)
                for partition in ("train", "validation", "test")
            },
        },
        "excluded_record_ids": sorted(excluded_ids),
        "excluded_by_reason": {
            reason: sorted(record_ids) for reason, record_ids in sorted(excluded.items())
        },
        "leakage_checks": {
            "content_group_cross_partition_count": sum(
                len(partitions) > 1 for partitions in group_partitions.values()
            ),
            "audio_target_cross_partition_count": sum(
                len(partitions) > 1 for partitions in target_partitions.values()
            ),
            "duplicate_window_id_count": len(windows)
            - len({window.window_id for window in windows}),
            "record_level_selected_plus_excluded_equals_input": (
                len(records_with_windows) + len(excluded_ids) == len(rows)
            ),
            "all_windows_exact_length": all(
                window.eeg_sample_count
                == round(window.window_sec * window.eeg_sampling_rate_hz)
                for window in windows
            ),
        },
        "deterministic_ordering": (
            "windows by window_id; exclusions and JSON keys lexicographically sorted"
        ),
    }
    return windows, audit


def write_pl_window_jsonl(
    path: str | Path,
    windows: Sequence[PLSpeechWindow],
) -> str:
    lines = [
        json.dumps(
            asdict(window),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for window in sorted(windows, key=lambda item: item.window_id)
    ]
    serialized = "\n".join(lines) + ("\n" if lines else "")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8", newline="\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_pl_window_jsonl(path: str | Path) -> list[PLSpeechWindow]:
    windows: list[PLSpeechWindow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("window_schema_version") != PL_WINDOW_SCHEMA_VERSION:
                raise ValueError(f"Unknown window schema at line {line_number}")
            windows.append(PLSpeechWindow(**payload))
    _validate_window_integrity(windows)
    return windows


def render_pl_window_audit_markdown(audit: Mapping[str, object]) -> str:
    counts = audit["counts"]
    leakage = audit["leakage_checks"]
    lines = [
        "# PL EEG–speech window audit",
        "",
        f"- Schema: `{audit['window_schema_version']}`",
        f"- Manifest PL records: {counts['input_pl_record_count']:,}",
        f"- Records with windows: {counts['records_with_windows']:,}",
        f"- Excluded records: {counts['excluded_record_count']:,}",
        f"- Windows: {counts['window_count']:,}",
        f"- Unique audio targets: {counts['audio_target_count']:,}",
        f"- Content groups: {counts['content_group_count']:,}",
        f"- Subject groups: {counts['subject_group_count']:,}",
        "",
        "## Partition counts",
        "",
        "| partition | windows |",
        "|---|---:|",
    ]
    for partition, count in counts["window_counts_by_partition"].items():
        lines.append(f"| {partition} | {count:,} |")
    lines.extend(
        [
            "",
            "## Exclusions",
            "",
            "| reason | records |",
            "|---|---:|",
        ]
    )
    for reason, record_ids in audit["excluded_by_reason"].items():
        lines.append(f"| `{reason}` | {len(record_ids):,} |")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Content groups crossing partitions: "
            f"{leakage['content_group_cross_partition_count']}",
            f"- Audio targets crossing partitions: "
            f"{leakage['audio_target_cross_partition_count']}",
            f"- Duplicate window IDs: {leakage['duplicate_window_id_count']}",
            f"- Complete record accounting: "
            f"`{leakage['record_level_selected_plus_excluded_equals_input']}`",
            f"- All windows exact length: `{leakage['all_windows_exact_length']}`",
            "",
            "Only verified audio-aligned PL trials are eligible. Other quality flags "
            "are recorded but are not used as filters.",
            "",
        ]
    )
    return "\n".join(lines)


def _windows_for_row(
    row: Mapping[str, object],
    *,
    split: str,
    spec: PLSpeechWindowSpec,
    audio_sample_rate_hz: int,
    audio_frame_count: int,
) -> list[PLSpeechWindow]:
    eeg_rate = int(round(float(row["eeg_sampling_rate"])))
    window_eeg_samples = round(spec.window_sec * eeg_rate)
    window_audio_samples = round(spec.window_sec * audio_sample_rate_hz)
    stride_eeg_samples = round(spec.stride_sec * eeg_rate)
    stride_audio_samples = round(spec.stride_sec * audio_sample_rate_hz)
    delay_eeg_samples = round(spec.delay_ms / 1000.0 * eeg_rate)
    audio_block_start = round(float(row["audio_start_sec"]) * audio_sample_rate_hz)
    audio_block_stop = round(float(row["audio_end_sec"]) * audio_sample_rate_hz)
    eeg_block_start = int(row["eeg_start_sample"])
    eeg_block_stop = int(row["eeg_end_sample"])
    if audio_block_stop > audio_frame_count:
        return []

    output: list[PLSpeechWindow] = []
    offset_index = 0
    while True:
        audio_start = audio_block_start + offset_index * stride_audio_samples
        audio_stop = audio_start + window_audio_samples
        eeg_start = (
            eeg_block_start + delay_eeg_samples + offset_index * stride_eeg_samples
        )
        eeg_stop = eeg_start + window_eeg_samples
        if (
            audio_start < audio_block_start
            or eeg_start < eeg_block_start
            or audio_stop > audio_block_stop
            or eeg_stop > eeg_block_stop
        ):
            break
        audio_target_id = _audio_target_id(
            row,
            audio_start_sample=audio_start,
            audio_stop_sample=audio_stop,
        )
        window_id = _window_id(
            record_id=str(row["record_id"]),
            audio_target_id=audio_target_id,
            eeg_start_sample=eeg_start,
            eeg_stop_sample=eeg_stop,
            delay_ms=spec.delay_ms,
        )
        output.append(
            PLSpeechWindow(
                window_schema_version=PL_WINDOW_SCHEMA_VERSION,
                window_id=window_id,
                record_id=str(row["record_id"]),
                block_id=str(row["block_id"]),
                split_group_id=str(row["split_group_id"]),
                split=split,
                subject_group_id=make_subject_group_id(row),
                subject_id=str(row["subject_id"]),
                session_id=str(row["session_id"]),
                run_id=str(row["run_id"]),
                speaker_id=str(row["speaker_id"]),
                stimulus_position=int(row["stimulus_position"]),
                char_count=(
                    int(row["char_count"])
                    if row["char_count"] is not None
                    else None
                ),
                eeg_file=str(row["eeg_file"]),
                eeg_sampling_rate_hz=eeg_rate,
                eeg_start_sample=eeg_start,
                eeg_stop_sample=eeg_stop,
                eeg_sample_count=window_eeg_samples,
                valid_eeg_samples=window_eeg_samples,
                padded_eeg_samples=0,
                audio_file=str(row["audio_file"]),
                audio_source_sample_rate_hz=audio_sample_rate_hz,
                audio_start_sample=audio_start,
                audio_stop_sample=audio_stop,
                audio_start_sec=audio_start / audio_sample_rate_hz,
                audio_stop_sec=audio_stop / audio_sample_rate_hz,
                audio_target_id=audio_target_id,
                window_offset_sec=offset_index * spec.stride_sec,
                window_sec=spec.window_sec,
                stride_sec=spec.stride_sec,
                eeg_delay_ms=spec.delay_ms,
                source_trial_eeg_duration_sec=float(row["eeg_duration_sec"]),
                overlap_source=str(row["audio_alignment_method"]),
                quality_flag=str(row["quality_flag"]),
            )
        )
        offset_index += 1
    return output


def _audio_target_id(
    row: Mapping[str, object],
    *,
    audio_start_sample: int,
    audio_stop_sample: int,
) -> str:
    path = Path(str(row["audio_file"]))
    source_identity = f"{path.parent.name}/{path.name}"
    payload = (
        f"pl-audio-target-v1\0{source_identity}\0"
        f"{audio_start_sample}\0{audio_stop_sample}"
    )
    return "pl-audio-target-v1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[
        :24
    ]


def _window_id(
    *,
    record_id: str,
    audio_target_id: str,
    eeg_start_sample: int,
    eeg_stop_sample: int,
    delay_ms: float,
) -> str:
    payload = (
        f"{PL_WINDOW_SCHEMA_VERSION}\0{record_id}\0{audio_target_id}\0"
        f"{eeg_start_sample}\0{eeg_stop_sample}\0{delay_ms:.9f}"
    )
    return "pl-window-v1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _soundfile_info(path: str) -> tuple[int, int]:
    import soundfile as sf

    info = sf.info(path)
    return int(info.samplerate), int(info.frames)


def _validate_window_integrity(windows: Sequence[PLSpeechWindow]) -> None:
    ids = [window.window_id for window in windows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate PL window_id")
    group_splits: dict[str, set[str]] = defaultdict(set)
    target_splits: dict[str, set[str]] = defaultdict(set)
    for window in windows:
        if window.eeg_stop_sample - window.eeg_start_sample != (
            window.eeg_sample_count
        ):
            raise ValueError(f"Invalid EEG sample count: {window.window_id}")
        if window.audio_stop_sample <= window.audio_start_sample:
            raise ValueError(f"Invalid audio range: {window.window_id}")
        group_splits[window.split_group_id].add(window.split)
        target_splits[window.audio_target_id].add(window.split)
    if any(len(values) > 1 for values in group_splits.values()):
        raise ValueError("A split_group_id crosses PL window partitions")
    if any(len(values) > 1 for values in target_splits.values()):
        raise ValueError("An audio_target_id crosses PL window partitions")
