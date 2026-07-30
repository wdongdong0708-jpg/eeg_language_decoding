"""Read-only audit of ChineseEEG EEG, text and audio assets."""

from __future__ import annotations

import configparser
import csv
import hashlib
import json
import math
import re
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

from data.readers.simple_xlsx import read_xlsx_sheet
from data.readers.stimulus_text import load_chineseeeg2_workbook
from data.text_normalization import normalize_text

_RUN_PATTERN = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)"
    r"_run-(?P<run>[^_]+)_eeg\.vhdr$"
)
_CHANNEL_PATTERN = re.compile(r"^ch([0-9]+)$", re.IGNORECASE)
_AUDIO_PATTERN = re.compile(r"^audio_([0-9]+)\.wav$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: str
    root: Path
    expected_sampling_rate_hz: float


def _read_config(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    text = path.read_text(encoding="utf-8-sig")
    first_section = text.find("[")
    if first_section < 0:
        raise ValueError(f"BrainVision file has no INI sections: {path}")
    parser.read_string(text[first_section:], source=str(path))
    return parser


def _parse_channels(
    parser: configparser.ConfigParser,
) -> tuple[list[str], list[float], list[str]]:
    entries: list[tuple[int, str, float, str]] = []
    for key, raw in parser["Channel Infos"].items():
        match = _CHANNEL_PATTERN.match(key)
        if match is None:
            continue
        fields = raw.split(",")
        name = fields[0]
        resolution = float(fields[2])
        unit = fields[3]
        entries.append((int(match.group(1)), name, resolution, unit))
    entries.sort()
    return (
        [item[1] for item in entries],
        [item[2] for item in entries],
        [item[3] for item in entries],
    )


def _read_events(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            events.append(
                {
                    "onset": float(row["onset"]),
                    "duration": float(row["duration"]),
                    "trial_type": row["trial_type"],
                    "value": row.get("value"),
                    "sample": int(row["sample"]),
                }
            )
    return events


def pair_row_trials(
    events: Iterable[dict[str, object]],
) -> tuple[list[tuple[int, int]], int, int]:
    """Pair ROWS/ROWE without crossing or fabricating boundaries."""

    pairs: list[tuple[int, int]] = []
    pending_start: int | None = None
    orphan_starts = 0
    orphan_ends = 0
    for event in sorted(events, key=lambda item: int(item["sample"])):
        event_type = str(event["trial_type"])
        sample = int(event["sample"])
        if event_type == "ROWS":
            if pending_start is not None:
                orphan_starts += 1
            pending_start = sample
        elif event_type == "ROWE":
            if pending_start is None:
                orphan_ends += 1
            elif sample <= pending_start:
                orphan_starts += 1
                orphan_ends += 1
                pending_start = None
            else:
                pairs.append((pending_start, sample))
                pending_start = None
    if pending_start is not None:
        orphan_starts += 1
    return pairs, orphan_starts, orphan_ends


def audit_brainvision_file(path: Path) -> dict[str, object]:
    parser = _read_config(path)
    common = parser["Common Infos"]
    binary = parser["Binary Infos"]
    number_of_channels = int(common["NumberOfChannels"])
    sampling_interval_us = float(common["SamplingInterval"])
    sampling_rate_hz = 1_000_000.0 / sampling_interval_us
    binary_format = binary["BinaryFormat"]
    bytes_per_value = {"IEEE_FLOAT_32": 4, "INT_16": 2, "UINT_16": 2}.get(
        binary_format
    )
    if bytes_per_value is None:
        raise ValueError(f"Unsupported BrainVision binary format: {binary_format}")

    referenced_data_path = path.parent / common["DataFile"]
    referenced_marker_path = path.parent / common["MarkerFile"]
    header_data_reference_exists = referenced_data_path.is_file()
    header_marker_reference_exists = referenced_marker_path.is_file()
    data_path = (
        referenced_data_path
        if header_data_reference_exists
        else path.with_suffix(".eeg")
    )
    marker_path = (
        referenced_marker_path
        if header_marker_reference_exists
        else path.with_suffix(".vmrk")
    )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Neither referenced nor same-stem BrainVision data file exists for {path}"
        )
    sample_width = number_of_channels * bytes_per_value
    data_bytes = data_path.stat().st_size
    data_size_remainder = data_bytes % sample_width
    n_times = data_bytes // sample_width
    channel_names, resolutions, source_units = _parse_channels(parser)
    events_path = path.with_name(path.name.replace("_eeg.vhdr", "_events.tsv"))
    events = _read_events(events_path)
    marker_counts = Counter(str(event["trial_type"]) for event in events)
    trials, orphan_starts, orphan_ends = pair_row_trials(events)
    match = _RUN_PATTERN.match(path.name)
    entities = match.groupdict() if match else {}

    marker_entry_count = 0
    if marker_path.is_file():
        with marker_path.open("r", encoding="utf-8-sig") as handle:
            marker_entry_count = sum(1 for line in handle if line.startswith("Mk"))

    return {
        "path": str(path),
        **entities,
        "format": {
            "header": ".vhdr",
            "data": ".eeg",
            "marker": ".vmrk",
            "binary_format": binary_format,
        },
        "sampling_rate_hz": sampling_rate_hz,
        "channel_count": number_of_channels,
        "channel_names": channel_names,
        "source_units": sorted(set(source_units)),
        "resolutions": sorted(set(resolutions)),
        "shape": [number_of_channels, n_times],
        "duration_sec": n_times / sampling_rate_hz,
        "data_size_remainder": data_size_remainder,
        "header_data_reference_exists": header_data_reference_exists,
        "header_marker_reference_exists": header_marker_reference_exists,
        "events_tsv_exists": events_path.is_file(),
        "event_count": len(events),
        "annotation_count_vmrk": marker_entry_count,
        "marker_counts": dict(sorted(marker_counts.items())),
        "row_trial_count": len(trials),
        "orphan_row_starts": orphan_starts,
        "orphan_row_ends": orphan_ends,
        "first_row_trial": list(trials[0]) if trials else None,
        "last_row_trial": list(trials[-1]) if trials else None,
    }


def _mne_representative(path: Path) -> dict[str, object]:
    try:
        import mne
    except ImportError as error:
        return {"path": str(path), "error": f"MNE unavailable: {error}"}

    try:
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    except Exception as error:  # MNE error type varies with malformed sidecars.
        return {"path": str(path), "error": repr(error)}
    stop = min(int(raw.n_times), int(raw.info["sfreq"]))
    data = raw.get_data(start=0, stop=stop)
    return {
        "path": str(path),
        "shape": [len(raw.ch_names), int(raw.n_times)],
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "channel_names": raw.ch_names,
        "source_units": sorted(set(raw._orig_units.values())),
        "mne_internal_unit": "V",
        "sample_dtype_after_mne": str(data.dtype),
        "first_second_min_v": float(data.min()),
        "first_second_max_v": float(data.max()),
        "annotation_count": len(raw.annotations),
        "annotation_descriptions": sorted(set(raw.annotations.description.tolist())),
    }


def audit_eeg_dataset(spec: DatasetSpec) -> dict[str, object]:
    files = sorted(spec.root.rglob("*_eeg.vhdr"))
    records = [audit_brainvision_file(path) for path in files]
    sampling_rates = sorted({record["sampling_rate_hz"] for record in records})
    channel_counts = sorted({record["channel_count"] for record in records})
    channel_layouts = {
        hashlib.sha256(
            "\0".join(record["channel_names"]).encode("utf-8")
        ).hexdigest()[:16]
        for record in records
    }
    durations = [float(record["duration_sec"]) for record in records]
    marker_totals: Counter[str] = Counter()
    for record in records:
        marker_totals.update(record["marker_counts"])

    anomalies: list[dict[str, object]] = []
    for record in records:
        reasons: list[str] = []
        if not math.isclose(
            float(record["sampling_rate_hz"]),
            spec.expected_sampling_rate_hz,
        ):
            reasons.append("unexpected_sampling_rate")
        if int(record["channel_count"]) != 128:
            reasons.append("unexpected_channel_count")
        if record["channel_names"] != [f"E{index}" for index in range(1, 129)]:
            reasons.append("unexpected_channel_names")
        if record["data_size_remainder"]:
            reasons.append("invalid_binary_shape")
        if not record["header_data_reference_exists"]:
            reasons.append("broken_header_data_reference")
        if not record["header_marker_reference_exists"]:
            reasons.append("broken_header_marker_reference")
        if not record["events_tsv_exists"]:
            reasons.append("missing_events_tsv")
        if record["orphan_row_starts"] or record["orphan_row_ends"]:
            reasons.append("unpaired_row_marker")
        if record["event_count"] != record["annotation_count_vmrk"]:
            reasons.append("events_vmrk_count_mismatch")
        if reasons:
            anomalies.append({"path": record["path"], "reasons": reasons})

    return {
        "dataset_id": spec.dataset_id,
        "root": str(spec.root),
        "recording_count": len(records),
        "format": "BrainVision triplet (.vhdr/.eeg/.vmrk) with BIDS TSV/JSON sidecars",
        "sampling_rates_hz": sampling_rates,
        "channel_counts": channel_counts,
        "channel_layout_fingerprints": sorted(channel_layouts),
        "channel_names": records[0]["channel_names"] if records else [],
        "source_units": sorted(
            {unit for record in records for unit in record["source_units"]}
        ),
        "resolutions": sorted(
            {value for record in records for value in record["resolutions"]}
        ),
        "binary_formats": sorted(
            {record["format"]["binary_format"] for record in records}
        ),
        "n_times_min": min((record["shape"][1] for record in records), default=0),
        "n_times_max": max((record["shape"][1] for record in records), default=0),
        "duration_sec_min": min(durations, default=0),
        "duration_sec_median": median(durations) if durations else 0,
        "duration_sec_max": max(durations, default=0),
        "event_marker_totals": dict(sorted(marker_totals.items())),
        "row_trial_count": sum(int(record["row_trial_count"]) for record in records),
        "missing_events_tsv_count": sum(
            not bool(record["events_tsv_exists"]) for record in records
        ),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "recordings": records,
        "representative": (
            _mne_representative(
                Path(
                    next(
                        record["path"]
                        for record in records
                        if record["header_data_reference_exists"]
                        and record["header_marker_reference_exists"]
                    )
                )
            )
            if any(
                record["header_data_reference_exists"]
                and record["header_marker_reference_exists"]
                for record in records
            )
            else None
        ),
    }


def _workbook_column_a(path: Path) -> list[str]:
    sheet = read_xlsx_sheet(path)
    return [
        str(row.values[0]).strip()
        for row in sheet.rows
        if row.excel_row > 1 and row.values and row.values[0] is not None
    ]


def _exact_row_correspondence(source: list[str], target: list[str]) -> dict[str, object]:
    source_normalized = [normalize_text(text) for text in source if normalize_text(text)]
    target_normalized = [normalize_text(text) for text in target if normalize_text(text)]
    source_set = set(source_normalized)
    target_set = set(target_normalized)
    matched_source = sum(item in target_set for item in source_normalized)
    matched_target = sum(item in source_set for item in target_normalized)
    return {
        "source_rows": len(source_normalized),
        "target_rows": len(target_normalized),
        "source_rows_with_exact_target": matched_source,
        "target_rows_with_exact_source": matched_target,
        "source_exact_match_rate": (
            matched_source / len(source_normalized) if source_normalized else 0
        ),
        "target_exact_match_rate": (
            matched_target / len(target_normalized) if target_normalized else 0
        ),
        "interpretation": (
            "Exact normalized row matches only; segmentation changes require "
            "sequence alignment."
        ),
    }


def audit_text(
    *,
    chineseeeg1_root: Path,
    chineseeeg2_materials_root: Path,
) -> dict[str, object]:
    ceeg2_original = chineseeeg2_materials_root / "original_novel"
    ceeg2_specs = {
        "littleprince": (ceeg2_original / "littleprince.xlsx", 19),
        "garnettdream": (ceeg2_original / "garnettdream.xlsx", 23),
    }
    ceeg2_units = {
        book: load_chineseeeg2_workbook(
            path,
            book_id=book,
            first_chapter_excel_row=first_chapter_row,
        )
        for book, (path, first_chapter_row) in ceeg2_specs.items()
    }

    ceeg1_novels = chineseeeg1_root / "novels"
    ceeg1_segmented = ceeg1_novels / "segmented_novel"
    ceeg1_rows: dict[str, list[str]] = {}
    ceeg1_runs: dict[str, list[dict[str, object]]] = {}
    for book_folder, book_id in (
        ("LittlePrince", "littleprince"),
        ("GarnettDream", "garnettdream"),
    ):
        folder = ceeg1_segmented / book_folder
        run_files = sorted(
            path
            for path in folder.glob("segmented_Chinense_novel_run_*.xlsx")
            if not path.stem.endswith("_display")
        )
        rows: list[str] = []
        run_summaries: list[dict[str, object]] = []
        for path in run_files:
            run_rows = _workbook_column_a(path)
            rows.extend(run_rows)
            run_summaries.append(
                {"path": str(path), "row_count": len(run_rows)}
            )
        ceeg1_rows[book_id] = rows
        ceeg1_runs[book_id] = run_summaries

    books: dict[str, object] = {}
    for book_id, units in ceeg2_units.items():
        chapter_rows = [
            {"excel_row": unit.excel_row, "chapter_id": unit.chapter_id}
            for unit in units
            if unit.is_chapter_marker
        ]
        f1_boundaries = [
            {"excel_row": unit.excel_row, "label": unit.f1_boundary_label}
            for unit in units
            if unit.f1_boundary_label
        ]
        m1_boundaries = [
            {"excel_row": unit.excel_row, "label": unit.m1_boundary_label}
            for unit in units
            if unit.m1_boundary_label
        ]
        books[book_id] = {
            "workbook": str(ceeg2_specs[book_id][0]),
            "first_chapter_excel_row": ceeg2_specs[book_id][1],
            "stimulus_row_count": len(units),
            "text_segment_count": sum(
                not unit.is_chapter_marker for unit in units
            ),
            "pre_chapter1_row_count": sum(
                unit.excel_row < ceeg2_specs[book_id][1] for unit in units
            ),
            "chapter_marker_rows": chapter_rows,
            "f1_boundary_rows": f1_boundaries,
            "m1_boundary_rows": m1_boundaries,
            "segment_id_example": next(
                (
                    unit.segment_id
                    for unit in units
                    if not unit.is_chapter_marker and unit.chapter_id >= 1
                ),
                None,
            ),
            "chineseeeg1_run_tables": ceeg1_runs[book_id],
            "cross_dataset_exact_row_correspondence": _exact_row_correspondence(
                ceeg1_rows[book_id],
                [unit.text for unit in units],
            ),
        }

    original_texts = sorted((ceeg1_novels / "original_novel").glob("*.txt"))
    return {
        "chineseeeg1_original_novels": [str(path) for path in original_texts],
        "chineseeeg2_books": books,
        "unit_definition": (
            "XLSX rows are stimulus sentence segments/display rows, not asserted "
            "linguistic sentences or spoken-word timestamps."
        ),
        "cross_paradigm_rule": (
            "ChineseEEG2 RA and PL share the row sequence only when speaker/material "
            "variant and event counts match; ChineseEEG1 requires canonical sequence alignment."
        ),
    }


def _audio_index(path: Path) -> int:
    match = _AUDIO_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected audio filename: {path.name}")
    return int(match.group(1))


def audit_audio(
    *,
    audio_root: Path,
    text_audit: dict[str, object],
    event_sampling_rate_hz: float = 250.0,
) -> dict[str, object]:
    groups: dict[str, object] = {}
    total_duration = 0.0
    sample_rates: set[int] = set()
    channel_counts: set[int] = set()

    expected_text_rows = {
        book: int(details["stimulus_row_count"])
        for book, details in text_audit["chineseeeg2_books"].items()
    }

    for folder in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        wav_files = sorted(folder.glob("audio_*.wav"), key=_audio_index)
        if not wav_files:
            continue
        wav_records: list[dict[str, object]] = []
        for path in wav_files:
            with wave.open(str(path), "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                frame_count = handle.getnframes()
                duration = frame_count / sample_rate
                sample_width_bytes = handle.getsampwidth()
            sample_rates.add(sample_rate)
            channel_counts.add(channels)
            total_duration += duration
            wav_records.append(
                {
                    "path": str(path),
                    "audio_index": _audio_index(path),
                    "sample_rate_hz": sample_rate,
                    "channels": channels,
                    "sample_width_bytes": sample_width_bytes,
                    "frame_count": frame_count,
                    "duration_sec": duration,
                }
            )

        book = "littleprince" if folder.name.startswith("littleprince") else "garnettdream"
        speaker = folder.name.rsplit("_", 1)[-1]
        events_path = folder / "events_data.json"
        alignment: dict[str, object] | None = None
        if events_path.is_file():
            data = json.loads(events_path.read_text(encoding="utf-8"))
            starts = [int(value) for value in data["begn_nonzero_indices"]]
            row_starts = [int(value) for value in data["ROWS_times"]]
            row_ends = [int(value) for value in data["ROWE_times"]]
            alignment = {
                "events_path": str(events_path),
                "audio_start_count": len(starts),
                "row_start_count": len(row_starts),
                "row_end_count": len(row_ends),
                "wav_count_matches_audio_starts": len(wav_records) == len(starts),
                "row_start_end_count_match": len(row_starts) == len(row_ends),
                "text_row_count": expected_text_rows[book],
                "row_start_count_matches_text_rows": (
                    len(row_starts) == expected_text_rows[book]
                ),
                "mapping_status": (
                    "validated_by_count"
                    if len(row_starts) == len(row_ends) == expected_text_rows[book]
                    else "requires_manual_or_sequence_alignment"
                ),
                "event_sampling_rate_hz": event_sampling_rate_hz,
            }

        durations = [float(record["duration_sec"]) for record in wav_records]
        groups[folder.name] = {
            "book_id": book,
            "speaker_id": speaker,
            "wav_count": len(wav_records),
            "audio_indices": [record["audio_index"] for record in wav_records],
            "sample_rates_hz": sorted({record["sample_rate_hz"] for record in wav_records}),
            "channel_counts": sorted({record["channels"] for record in wav_records}),
            "duration_sec_min": min(durations),
            "duration_sec_median": median(durations),
            "duration_sec_max": max(durations),
            "duration_sec_total": sum(durations),
            "files": wav_records,
            "alignment": alignment,
        }

    return {
        "root": str(audio_root),
        "wav_count": sum(int(group["wav_count"]) for group in groups.values()),
        "sample_rates_hz": sorted(sample_rates),
        "channel_counts": sorted(channel_counts),
        "duration_sec_total": total_duration,
        "groups": groups,
        "pl_subject_to_speaker": {
            **{f"sub-{index:02d}": "f1" for index in range(1, 5)},
            **{f"sub-{index:02d}": "m1" for index in range(5, 9)},
        },
    }


def build_full_audit(
    *,
    chineseeeg1_root: Path,
    chineseeeg2_pl_root: Path,
    chineseeeg2_ra_root: Path,
    chineseeeg1_derivatives_root: Path,
    chineseeeg2_materials_root: Path,
) -> dict[str, object]:
    eeg = {
        spec.dataset_id: audit_eeg_dataset(spec)
        for spec in (
            DatasetSpec("chineseeeg1", chineseeeg1_root, 256.0),
            DatasetSpec("chineseeeg2_pl", chineseeeg2_pl_root, 250.0),
            DatasetSpec("chineseeeg2_ra", chineseeeg2_ra_root, 250.0),
        )
    }
    text = audit_text(
        chineseeeg1_root=chineseeeg1_derivatives_root,
        chineseeeg2_materials_root=chineseeeg2_materials_root,
    )
    audio = audit_audio(
        audio_root=chineseeeg2_materials_root / "audio",
        text_audit=text,
    )
    return {
        "schema_version": "data-audit-v1",
        "eeg": eeg,
        "text": text,
        "audio": audio,
    }


def audit_to_markdown(audit: dict[str, object]) -> str:
    lines = [
        "# ChineseEEG data audit",
        "",
        "Generated by `scripts/audit_data.py`; all operations are read-only.",
        "The companion JSON contains one record per EEG file and per WAV file.",
        "",
        "## EEG",
        "",
    ]
    for dataset_id, summary in audit["eeg"].items():
        anomaly_reasons: Counter[str] = Counter()
        for anomaly in summary["anomalies"]:
            anomaly_reasons.update(anomaly["reasons"])
        representative = summary["representative"]
        lines.extend(
            [
                f"### {dataset_id}",
                "",
                f"- Recordings: {summary['recording_count']}",
                f"- Format: {summary['format']}",
                f"- Sampling rates: {summary['sampling_rates_hz']} Hz",
                f"- Channels: {summary['channel_counts']}; names E1–E128",
                f"- Source unit: {summary['source_units']}; MNE converts values to V",
                f"- Binary format: {summary['binary_formats']}",
                f"- Shape range: 128 × {summary['n_times_min']} to "
                f"128 × {summary['n_times_max']}",
                f"- Duration range: {summary['duration_sec_min']:.3f} to "
                f"{summary['duration_sec_max']:.3f} s",
                f"- Event marker totals: "
                f"`{json.dumps(summary['event_marker_totals'], ensure_ascii=False)}`",
                "- Trial boundary rule: pair each `ROWS` with the next valid `ROWE`; "
                "never fabricate a missing boundary or cross a nested `ROWS`",
                f"- Paired ROWS/ROWE trials: {summary['row_trial_count']}",
                f"- Anomalous recordings: {summary['anomaly_count']}",
                f"- Anomaly reason counts: "
                f"`{json.dumps(dict(sorted(anomaly_reasons.items())), ensure_ascii=False)}`",
                (
                    "- Representative MNE read: "
                    f"shape {representative['shape']}, dtype "
                    f"`{representative['sample_dtype_after_mne']}`, "
                    f"{representative['annotation_count']} annotations, descriptions "
                    f"`{representative['annotation_descriptions']}`"
                    if representative and "error" not in representative
                    else f"- Representative MNE read: `{representative}`"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "ChineseEEG1 note: the 172 broken header references are GarnettDream "
            "headers whose `DataFile`/`MarkerFile` values spell the session "
            "`GranettDream`. Same-stem `.eeg`/`.vmrk` files exist. The auditor uses "
            "those files only for read-only inspection and retains the header defect "
            "as an anomaly; source data are not repaired in place.",
            "",
        ]
    )

    lines.extend(["## Text", ""])
    for book, details in audit["text"]["chineseeeg2_books"].items():
        correspondence_json = json.dumps(
            details["cross_dataset_exact_row_correspondence"],
            ensure_ascii=False,
        )
        lines.extend(
            [
                f"### {book}",
                "",
                f"- Workbook: `{details['workbook']}`",
                f"- Non-empty data rows (including chapter markers): "
                f"{details['stimulus_row_count']}",
                f"- Text/display segments (excluding chapter markers): "
                f"{details['text_segment_count']}",
                f"- First Ch1 marker Excel row: {details['first_chapter_excel_row']}",
                f"- Rows retained before Ch1 as explicit chapter 0/preface material: "
                f"{details['pre_chapter1_row_count']}",
                f"- Chapter markers: {len(details['chapter_marker_rows'])}",
                f"- f1/m1 boundary markers: {len(details['f1_boundary_rows'])}/"
                f"{len(details['m1_boundary_rows'])}",
                f"- Source segment ID example: `{details['segment_id_example']}`; "
                "final cross-dataset `content_id` is withheld until reviewed "
                "sequence alignment",
                f"- Cross-dataset exact-row audit: `{correspondence_json}`",
                "",
            ]
        )

    lines.extend(
        [
            "XLSX rows are treated as stimulus sentence segments/display rows. "
            "They are not asserted linguistic sentences or spoken-word timestamps.",
            "",
            "## Audio",
            "",
            f"- Root: `{audit['audio']['root']}`",
            f"- WAV files: {audit['audio']['wav_count']}",
            f"- Sample rates: {audit['audio']['sample_rates_hz']} Hz",
            f"- Channel counts: {audit['audio']['channel_counts']}",
            f"- Total duration: {audit['audio']['duration_sec_total']:.3f} s",
            "",
            "### Audio/text mapping status",
            "",
            "| group | wav | duration total/range (s) | row starts | row ends | "
            "text rows | status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for group_name, group in audit["audio"]["groups"].items():
        alignment = group["alignment"]
        if alignment is None:
            values = ["n/a", "n/a", "n/a", "no events_data.json"]
        else:
            values = [
                alignment["row_start_count"],
                alignment["row_end_count"],
                alignment["text_row_count"],
                alignment["mapping_status"],
            ]
        lines.append(
            f"| {group_name} | {group['wav_count']} | "
            f"{group['duration_sec_total']:.3f} / "
            f"{group['duration_sec_min']:.3f}–{group['duration_sec_max']:.3f} | "
            f"{values[0]} | {values[1]} | {values[2]} | {values[3]} |"
        )
    lines.extend(
        [
            "",
            "Only count-validated row mappings may be joined directly. Other groups "
            "require reviewed sequence or forced alignment.",
            "A count match is a necessary structural check, not proof of semantic "
            "alignment; downstream manifests must preserve the alignment method and "
            "provenance.",
            "",
        ]
    )
    return "\n".join(lines)
