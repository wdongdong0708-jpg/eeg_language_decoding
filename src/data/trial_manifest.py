"""Deterministic construction of the complete stimulus-row trial manifest."""

from __future__ import annotations

import bisect
import configparser
import csv
import hashlib
import json
import math
import re
import statistics
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from data.manifest import (
    BLOCK_ID_VERSION,
    EEG_END_SAMPLE_SEMANTICS,
    GLOBAL_TEXT_ID_VERSION,
    MANIFEST_SCHEMA_VERSION,
    RECORD_ID_VERSION,
    SENTENCE_ID_VERSION,
    SPLIT_GROUP_ID_VERSION,
    ManifestRecord,
    QualityFlag,
    encode_quality_flags,
    manifest_arrow_schema,
)
from data.readers.simple_xlsx import read_xlsx_sheet
from data.readers.stimulus_text import StimulusTextUnit, load_chineseeeg2_workbook
from data.splitting import assign_split
from data.text_alignment import (
    ALIGNMENT_VERSION,
    AlignmentTextUnit,
    align_monotonic_text_sequences,
    ensure_override_file,
    load_overrides,
)
from data.text_normalization import (
    CHAR_COUNT_METHOD,
    CONTENT_ID_VERSION,
    NORMALIZATION_VERSION,
    RAW_CHAR_COUNT_METHOD,
    TEXT_HASH_ALGORITHM,
    deterministic_word_count,
    highlight_char_count,
    jieba_word_count_method,
    make_content_id,
    non_whitespace_char_count,
    normalize_identifier,
    normalize_text_with_trace,
    raw_char_count,
)

DEFAULT_SPLIT_SEED = 20260730
EVENT_ALIGNMENT_VERSION = "event-to-stimulus-row-v1"
AUDIO_ALIGNMENT_VERSION = "pl-audio-events-v1"

_HEADER_PATTERN = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)"
    r"_run-(?P<run>[^_]+)_eeg\.vhdr$",
    re.IGNORECASE,
)
_CHAPTER_EVENT_PATTERN = re.compile(r"^CH\s*0*([0-9]+)$", re.IGNORECASE)
_CHAPTER_TEXT_PATTERN = re.compile(r"^[0-9]+$")


@dataclass(frozen=True, slots=True)
class ManifestPaths:
    chineseeeg1_eeg_root: Path
    chineseeeg1_novel_root: Path
    chineseeeg2_pl_root: Path
    chineseeeg2_ra_root: Path
    chineseeeg2_material_root: Path
    chineseeeg2_audio_root: Path
    audit_json: Path


@dataclass(frozen=True, slots=True)
class EventPair:
    start_sample: int
    end_sample: int
    pair_index: int
    preceding_chapter_event: str | None


@dataclass(frozen=True, slots=True)
class EventParseResult:
    pairs: tuple[EventPair, ...]
    orphan_starts: int
    orphan_ends: int
    first_chapter_sample: int | None


@dataclass(frozen=True, slots=True)
class MaterialUnit:
    book_id: str
    sentence_id: str
    global_text_id: str
    raw_text: str
    normalized_text: str
    split_identity_normalized_text: str
    chapter_id: int
    is_chapter_marker: bool
    source_excel_file: str
    source_excel_row: int
    normalization_trace: str
    raw_text_hash: str
    normalized_text_hash: str
    global_alignment_status: str
    global_alignment_score: float | None
    global_alignment_evidence: str


@dataclass(frozen=True, slots=True)
class UnitMatch:
    unit: MaterialUnit | None
    status: str
    score: float | None
    evidence: str


@dataclass(frozen=True, slots=True)
class AudioSpan:
    audio_file: str
    start_sec: float
    end_sec: float
    evidence: str


@dataclass(frozen=True, slots=True)
class RecordingSpec:
    dataset_version: str
    paradigm: str
    sampling_rate: float
    eeg_root: Path


def default_manifest_paths(
    *,
    audit_json: str | Path = "reports/data_audit.json",
) -> ManifestPaths:
    audit_path = Path(audit_json)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    eeg = audit["eeg"]
    ce1_root = Path(eeg["chineseeeg1"]["root"])
    ce1_novel_root = Path(
        audit["text"]["chineseeeg1_original_novels"][0]
    ).parent.parent
    ce2_material_root = Path(
        audit["text"]["chineseeeg2_books"]["littleprince"]["workbook"]
    ).parent.parent
    return ManifestPaths(
        chineseeeg1_eeg_root=ce1_root,
        chineseeeg1_novel_root=ce1_novel_root,
        chineseeeg2_pl_root=Path(eeg["chineseeeg2_pl"]["root"]),
        chineseeeg2_ra_root=Path(eeg["chineseeeg2_ra"]["root"]),
        chineseeeg2_material_root=ce2_material_root,
        chineseeeg2_audio_root=Path(audit["audio"]["root"]),
        audit_json=audit_path,
    )


def _stable_id(version: str, *parts: object, length: int = 24) -> str:
    payload = json.dumps(
        [version, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{version}-{digest[:length]}"


def _natural_run_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


def read_event_pairs(path: str | Path) -> EventParseResult:
    """Pair only legal ROWS then ROWE markers using [start,end) samples."""

    event_path = Path(path)
    if not event_path.is_file():
        return EventParseResult((), 0, 0, None)
    events: list[tuple[int, int, str]] = []
    with event_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for order, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            events.append((int(row["sample"]), order, row["trial_type"].strip()))
    events.sort(key=lambda item: (item[0], item[1]))

    pending_start: int | None = None
    preceding_chapter: str | None = None
    first_chapter_sample: int | None = None
    pairs: list[EventPair] = []
    orphan_starts = 0
    orphan_ends = 0
    for sample, _, trial_type in events:
        if _CHAPTER_EVENT_PATTERN.fullmatch(trial_type):
            preceding_chapter = trial_type
            if first_chapter_sample is None:
                first_chapter_sample = sample
        elif trial_type == "ROWS":
            if pending_start is not None:
                orphan_starts += 1
            pending_start = sample
        elif trial_type == "ROWE":
            if pending_start is None:
                orphan_ends += 1
            elif sample <= pending_start:
                orphan_starts += 1
                orphan_ends += 1
                pending_start = None
            else:
                pairs.append(
                    EventPair(
                        start_sample=pending_start,
                        end_sample=sample,
                        pair_index=len(pairs),
                        preceding_chapter_event=preceding_chapter,
                    )
                )
                pending_start = None
    if pending_start is not None:
        orphan_starts += 1
    return EventParseResult(
        pairs=tuple(pairs),
        orphan_starts=orphan_starts,
        orphan_ends=orphan_ends,
        first_chapter_sample=first_chapter_sample,
    )


def _raw_cell_text(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _ce2_global_text_id(book_id: str, excel_row: int, normalized_text: str) -> str:
    return _stable_id(
        GLOBAL_TEXT_ID_VERSION,
        "chineseeeg2-canonical-workbook",
        book_id,
        excel_row,
        normalized_text,
    )


def _material_unit_from_ce2(unit: StimulusTextUnit, workbook: Path) -> MaterialUnit:
    normalized = normalize_text_with_trace(unit.text)
    return MaterialUnit(
        book_id=unit.book_id,
        sentence_id=(
            f"{SENTENCE_ID_VERSION}-ce2-{unit.book_id}-xlsx{unit.excel_row:06d}"
        ),
        global_text_id=_ce2_global_text_id(
            unit.book_id, unit.excel_row, normalized.normalized_text
        ),
        raw_text=unit.text,
        normalized_text=normalized.normalized_text,
        split_identity_normalized_text=normalized.normalized_text,
        chapter_id=unit.chapter_id,
        is_chapter_marker=unit.is_chapter_marker,
        source_excel_file=str(workbook.resolve()),
        source_excel_row=unit.excel_row,
        normalization_trace=normalized.trace_json,
        raw_text_hash=normalized.raw_text_hash,
        normalized_text_hash=normalized.normalized_text_hash,
        global_alignment_status="exact",
        global_alignment_score=100.0,
        global_alignment_evidence="canonical ChineseEEG2 workbook row",
    )


def load_ce2_catalog(
    material_root: str | Path,
) -> dict[str, list[MaterialUnit]]:
    original = Path(material_root) / "original_novel"
    specifications = {
        "littleprince": (original / "littleprince.xlsx", 19),
        "garnettdream": (original / "garnettdream.xlsx", 23),
    }
    catalog: dict[str, list[MaterialUnit]] = {}
    for book_id, (workbook, first_chapter_row) in specifications.items():
        units = load_chineseeeg2_workbook(
            workbook,
            book_id=book_id,
            first_chapter_excel_row=first_chapter_row,
        )
        catalog[book_id] = [
            _material_unit_from_ce2(unit, workbook) for unit in units
        ]
    return catalog


def _ce1_run_workbooks(novel_root: Path, book_id: str) -> list[Path]:
    folder_name = "LittlePrince" if book_id == "littleprince" else "GarnettDream"
    folder = novel_root / "segmented_novel" / folder_name
    workbooks = [
        path
        for path in folder.glob("segmented_Chinense_novel_run_*.xlsx")
        if not path.stem.endswith("_display")
    ]
    return sorted(
        workbooks,
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )


def load_ce1_catalog(
    novel_root: str | Path,
    ce2_catalog: dict[str, list[MaterialUnit]],
    *,
    override_path: str | Path,
) -> tuple[dict[tuple[str, str], list[MaterialUnit]], dict[str, Counter[str]]]:
    root = Path(novel_root)
    ensure_override_file(override_path)
    overrides = load_overrides(override_path)
    result: dict[tuple[str, str], list[MaterialUnit]] = {}
    status_counts: dict[str, Counter[str]] = {}

    for book_id in ("littleprince", "garnettdream"):
        provisional: list[tuple[str, str, int, str, object]] = []
        for workbook in _ce1_run_workbooks(root, book_id):
            run_id = str(int(workbook.stem.rsplit("_", 1)[-1]))
            for row in read_xlsx_sheet(workbook).rows:
                if row.excel_row == 1 or not row.values or row.values[0] is None:
                    continue
                raw_text = _raw_cell_text(row.values[0])
                if not raw_text.strip():
                    continue
                normalized = normalize_text_with_trace(raw_text)
                sentence_id = (
                    f"{SENTENCE_ID_VERSION}-ce1-{book_id}-"
                    f"run{int(run_id):02d}-xlsx{row.excel_row:06d}"
                )
                provisional.append(
                    (run_id, sentence_id, row.excel_row, str(workbook.resolve()), normalized)
                )

        source_alignment_units = [
            AlignmentTextUnit(
                sentence_id=sentence_id,
                raw_text=normalization.raw_text,
                normalized_text=normalization.normalized_text,
            )
            for _, sentence_id, _, _, normalization in provisional
        ]
        target_alignment_units = [
            AlignmentTextUnit(
                sentence_id=unit.sentence_id,
                raw_text=unit.raw_text,
                normalized_text=unit.normalized_text,
                global_text_id=unit.global_text_id,
            )
            for unit in ce2_catalog[book_id]
        ]
        alignments = align_monotonic_text_sequences(
            source_alignment_units,
            target_alignment_units,
            overrides=overrides,
        )
        target_by_global = {
            unit.global_text_id: unit for unit in ce2_catalog[book_id]
        }
        counts: Counter[str] = Counter()
        for run_id, sentence_id, excel_row, workbook, normalization in provisional:
            alignment = alignments[sentence_id]
            counts[alignment.status] += 1
            target = (
                target_by_global.get(alignment.global_text_id)
                if alignment.global_text_id
                else None
            )
            global_text_id = alignment.global_text_id or _stable_id(
                GLOBAL_TEXT_ID_VERSION,
                "chineseeeg1-local-unresolved",
                sentence_id,
                normalization.normalized_text,
            )
            chapter_id = target.chapter_id if target is not None else 0
            is_marker = bool(
                _CHAPTER_TEXT_PATTERN.fullmatch(normalization.raw_text.strip())
            )
            result.setdefault((book_id, run_id), []).append(
                MaterialUnit(
                    book_id=book_id,
                    sentence_id=sentence_id,
                    global_text_id=global_text_id,
                    raw_text=normalization.raw_text,
                    normalized_text=normalization.normalized_text,
                    split_identity_normalized_text=(
                        target.normalized_text
                        if target is not None
                        else normalization.normalized_text
                    ),
                    chapter_id=chapter_id,
                    is_chapter_marker=is_marker,
                    source_excel_file=workbook,
                    source_excel_row=excel_row,
                    normalization_trace=normalization.trace_json,
                    raw_text_hash=normalization.raw_text_hash,
                    normalized_text_hash=normalization.normalized_text_hash,
                    global_alignment_status=alignment.status,
                    global_alignment_score=alignment.score,
                    global_alignment_evidence=alignment.evidence,
                )
            )
        status_counts[book_id] = counts
    return result, status_counts


def _units_by_excel_range(
    units: Sequence[MaterialUnit],
    start: int,
    end: int,
) -> list[MaterialUnit]:
    by_row = {unit.source_excel_row: unit for unit in units}
    missing = [row for row in range(start, end + 1) if row not in by_row]
    if missing:
        raise ValueError(f"Canonical workbook range has missing rows: {missing[:5]}")
    return [by_row[row] for row in range(start, end + 1)]


def _littleprince_chapter(run_id: str) -> int | None:
    if run_id.startswith("1") and len(run_id) >= 2:
        return int(run_id[1:])
    if run_id.startswith("2") and len(run_id) >= 2:
        return 14 + int(run_id[1:])
    return None


def select_ce2_run_units(
    *,
    book_id: str,
    run_id: str,
    subject_id: str,
    paradigm: str,
    catalog: dict[str, list[MaterialUnit]],
) -> tuple[list[MaterialUnit], str, str]:
    """Return evidence-backed material sequence for a CE2 recording."""

    units = catalog[book_id]
    if book_id == "littleprince":
        chapter = _littleprince_chapter(run_id)
        if chapter is None:
            return [], "unknown", "unrecognized LittlePrince encoded run"
        selected = [unit for unit in units if unit.chapter_id == chapter]
        return (
            selected,
            "f1" if subject_id in {"01", "02", "03", "04", "f1", "f2"} else "m1",
            "official encoded run to chapter map; canonical workbook chapter sequence",
        )

    if paradigm == "passive_listening":
        variant = "f1" if subject_id in {"01", "02", "03", "04"} else "m1"
    else:
        variant = subject_id

    # Boundaries come from the workbook f1/m1 columns and retain the documented
    # overlap at Excel row 1320 in the f1 stimulus variant.
    f1_ranges = {
        "11": (22, 253),
        "12": (254, 516),
        "13": (517, 685),
        "14": (686, 1026),
        "15": (1027, 1320),
        "21": (1320, 1516),
        "22": (1517, 1798),
        "23": (1799, 2033),
        "24": (2034, 2167),
    }
    m1_pl_ranges = {
        "11": (23, 516),
        "12": (517, 685),
        "13": (686, 1026),
        "14": (1027, 1516),
        "21": (1517, 1798),
        "22": (1799, 2033),
        "23": (2034, 2327),
        "24": (2328, 2513),
    }
    m1_ra_ranges = {
        "11": (23, 516),
        "12": (517, 685),
        "13": (686, 1026),
        "14": (1027, 1516),
        "15": (1517, 1798),
        "21": (1799, 2033),
        "22": (2034, 2327),
        "23": (2328, 2513),
    }
    ranges = (
        m1_ra_ranges
        if paradigm == "reading_aloud" and variant == "m1"
        else m1_pl_ranges
        if variant == "m1"
        else f1_ranges
    )
    selected_range = ranges.get(run_id)
    if selected_range is None:
        return [], variant, "no reviewed material range for run"
    if paradigm == "reading_aloud" and variant == "f1" and run_id == "24":
        selected_range = (2034, 2463)
    selected = _units_by_excel_range(units, *selected_range)
    return (
        selected,
        variant,
        (
            "official workbook material-variant boundary columns plus ordered "
            "duration-signature validation"
        ),
    )


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _duration_validation(
    pairs: Sequence[EventPair],
    units: Sequence[MaterialUnit],
    *,
    sampling_rate: float,
    pace_sec: float,
) -> tuple[bool, float | None, str]:
    actual = [
        (pair.end_sample - pair.start_sample) / sampling_rate for pair in pairs
    ]
    expected = [highlight_char_count(unit.raw_text) * pace_sec for unit in units]
    correlation = _pearson(actual, expected)
    residuals = [abs(a - b) for a, b in zip(actual, expected)]
    median_residual = statistics.median(residuals) if residuals else math.inf
    if len(actual) < 3:
        valid = len(actual) == len(expected) and median_residual <= 0.75
    else:
        valid = correlation is not None and correlation >= 0.90
    evidence = json.dumps(
        {
            "version": EVENT_ALIGNMENT_VERSION,
            "row_count": len(units),
            "event_pair_count": len(pairs),
            "fixed_visual_pace_sec": pace_sec,
            "duration_char_count_pearson": correlation,
            "median_absolute_duration_residual_sec": median_residual,
            "interpretation": "sequence evidence, not speech timestamps",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return valid, correlation, evidence


def _sequence_align_by_duration(
    pairs: Sequence[EventPair],
    units: Sequence[MaterialUnit],
    *,
    sampling_rate: float,
    pace_sec: float,
) -> list[UnitMatch]:
    """Monotonic DP for count-mismatched event/unit sequences; never zip-truncates."""

    n_pairs = len(pairs)
    n_units = len(units)
    if not n_pairs:
        return []
    if not n_units:
        return [
            UnitMatch(None, "unresolved", None, f"{EVENT_ALIGNMENT_VERSION}:no_units")
            for _ in pairs
        ]
    actual = [
        (pair.end_sample - pair.start_sample) / sampling_rate for pair in pairs
    ]
    expected = [highlight_char_count(unit.raw_text) * pace_sec for unit in units]
    offset = statistics.median(actual) - statistics.median(expected)
    scale = max(0.15, pace_sec)
    gap_cost = 2.0
    infinity = float("inf")
    costs = [[infinity] * (n_units + 1) for _ in range(n_pairs + 1)]
    back = [[""] * (n_units + 1) for _ in range(n_pairs + 1)]
    costs[0][0] = 0.0
    for i in range(1, n_pairs + 1):
        costs[i][0] = i * gap_cost
        back[i][0] = "event_gap"
    for j in range(1, n_units + 1):
        costs[0][j] = j * gap_cost
        back[0][j] = "unit_gap"
    for i in range(1, n_pairs + 1):
        for j in range(1, n_units + 1):
            residual = abs(actual[i - 1] - (expected[j - 1] + offset))
            match_cost = min(residual / scale, gap_cost * 2.5)
            options = (
                (costs[i - 1][j - 1] + match_cost, "match"),
                (costs[i - 1][j] + gap_cost, "event_gap"),
                (costs[i][j - 1] + gap_cost, "unit_gap"),
            )
            costs[i][j], back[i][j] = min(options, key=lambda item: item[0])

    mapped: list[int | None] = [None] * n_pairs
    residual_by_pair: list[float | None] = [None] * n_pairs
    i, j = n_pairs, n_units
    while i or j:
        operation = back[i][j]
        if operation == "match":
            mapped[i - 1] = j - 1
            residual_by_pair[i - 1] = abs(
                actual[i - 1] - (expected[j - 1] + offset)
            )
            i -= 1
            j -= 1
        elif operation == "event_gap":
            i -= 1
        elif operation == "unit_gap":
            j -= 1
        else:
            break

    output: list[UnitMatch] = []
    for unit_index, residual in zip(mapped, residual_by_pair):
        if unit_index is None or residual is None or residual > max(0.8, pace_sec * 2):
            output.append(
                UnitMatch(
                    None,
                    "unresolved",
                    None,
                    f"{EVENT_ALIGNMENT_VERSION}:duration_dp_rejected",
                )
            )
        else:
            score = max(0.0, 100.0 * (1.0 - residual / max(0.8, pace_sec * 2)))
            output.append(
                UnitMatch(
                    units[unit_index],
                    "fuzzy",
                    score,
                    f"{EVENT_ALIGNMENT_VERSION}:monotonic_duration_dp",
                )
            )
    return output


def align_event_pairs(
    pairs: Sequence[EventPair],
    units: Sequence[MaterialUnit],
    *,
    sampling_rate: float,
    pace_sec: float,
    base_evidence: str,
) -> list[UnitMatch]:
    if len(pairs) == len(units):
        valid, correlation, evidence = _duration_validation(
            pairs,
            units,
            sampling_rate=sampling_rate,
            pace_sec=pace_sec,
        )
        if valid:
            score = 100.0 if correlation is None else max(0.0, correlation * 100.0)
            return [
                UnitMatch(
                    unit,
                    "exact",
                    score,
                    f"{base_evidence};{evidence}",
                )
                for unit in units
            ]
    return _sequence_align_by_duration(
        pairs,
        units,
        sampling_rate=sampling_rate,
        pace_sec=pace_sec,
    )


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def load_validated_littleprince_audio_spans(
    audio_root: str | Path,
    ce2_catalog: dict[str, list[MaterialUnit]],
) -> tuple[dict[tuple[str, str], AudioSpan], dict[str, object]]:
    """Validate PL audio evidence and map canonical row IDs to half-open spans."""

    root = Path(audio_root)
    spans: dict[tuple[str, str], AudioSpan] = {}
    diagnostics: dict[str, object] = {}
    units = ce2_catalog["littleprince"]
    for speaker in ("f1", "m1"):
        folder = root / f"littleprince_{speaker}"
        events_path = folder / "events_data.json"
        data = json.loads(events_path.read_text(encoding="utf-8"))
        starts = [int(value) for value in data["begn_nonzero_indices"]]
        row_starts = [int(value) for value in data["ROWS_times"]]
        row_ends = [int(value) for value in data["ROWE_times"]]
        wavs = sorted(
            folder.glob("audio_*.wav"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        durations = [_wav_duration(path) for path in wavs]
        actual = [(end - start) / 250.0 for start, end in zip(row_starts, row_ends)]
        expected = [highlight_char_count(unit.raw_text) * 0.25 for unit in units]
        correlation = _pearson(actual, expected)
        residuals = [abs(a - b) for a, b in zip(actual, expected)]
        validated = (
            len(starts) == len(wavs)
            and len(row_starts) == len(row_ends) == len(units)
            and correlation is not None
            and correlation >= 0.99
            and statistics.quantiles(residuals, n=20)[18] <= 0.08
        )
        diagnostics[speaker] = {
            "events_data": str(events_path.resolve()),
            "wav_count": len(wavs),
            "audio_start_count": len(starts),
            "row_count": len(row_starts),
            "text_unit_count": len(units),
            "duration_char_count_pearson": correlation,
            "duration_residual_p95_sec": statistics.quantiles(residuals, n=20)[18],
            "validated": validated,
        }
        if not validated:
            continue
        for unit, row_start, row_end in zip(units, row_starts, row_ends):
            audio_index = bisect.bisect_right(starts, row_start) - 1
            if not 0 <= audio_index < len(wavs):
                continue
            start_sec = (row_start - starts[audio_index]) / 250.0
            end_sec = (row_end - starts[audio_index]) / 250.0
            if start_sec < 0 or end_sec <= start_sec or end_sec > durations[audio_index] + 0.1:
                continue
            evidence = json.dumps(
                {
                    "version": AUDIO_ALIGNMENT_VERSION,
                    "events_data": str(events_path.resolve()),
                    "speaker": speaker,
                    "canonical_excel_row": unit.source_excel_row,
                    "duration_char_count_pearson": correlation,
                    "file_index": audio_index,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            spans[(speaker, unit.global_text_id)] = AudioSpan(
                audio_file=str(wavs[audio_index].resolve()),
                start_sec=start_sec,
                end_sec=end_sec,
                evidence=evidence,
            )
    return spans, diagnostics


def _recording_anomaly_flags(
    audit: dict[str, object],
    dataset_key: str,
) -> dict[str, set[QualityFlag]]:
    flags: dict[str, set[QualityFlag]] = defaultdict(set)
    mapping = {
        "unpaired_row_marker": QualityFlag.ORPHAN_ROW_EVENT_IN_RECORDING,
        "missing_events_tsv": QualityFlag.MISSING_EVENTS_TSV,
        "events_vmrk_count_mismatch": QualityFlag.EVENTS_VMRK_COUNT_MISMATCH,
        "broken_header_data_reference": QualityFlag.BROKEN_BRAINVISION_REFERENCE,
        "broken_header_marker_reference": QualityFlag.BROKEN_BRAINVISION_REFERENCE,
        "invalid_binary_shape": QualityFlag.INVALID_BRAINVISION_BINARY,
    }
    for anomaly in audit["eeg"][dataset_key]["anomalies"]:
        path = str(Path(anomaly["path"]).resolve())
        for reason in anomaly["reasons"]:
            if reason in mapping:
                flags[path].add(mapping[reason])
    return flags


def _brainvision_references(header_path: Path) -> tuple[Path, Path]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    text = header_path.read_text(encoding="utf-8-sig")
    parser.read_string(text[text.find("[") :])
    common = parser["Common Infos"]
    referenced_data = header_path.parent / common["DataFile"]
    data_path = referenced_data if referenced_data.is_file() else header_path.with_suffix(".eeg")
    return data_path.resolve(), header_path.resolve()


def _manifest_text_fields(unit: MaterialUnit | None) -> dict[str, object]:
    if unit is None:
        return {
            "raw_text": None,
            "normalized_text": None,
            "text_hash": None,
            "raw_text_hash": None,
            "normalized_text_hash": None,
            "char_count": None,
            "raw_char_count": None,
            "highlight_char_count": None,
            "word_count": None,
            "normalization_trace": "[]",
        }
    return {
        "raw_text": unit.raw_text,
        "normalized_text": unit.normalized_text,
        "text_hash": unit.normalized_text_hash,
        "raw_text_hash": unit.raw_text_hash,
        "normalized_text_hash": unit.normalized_text_hash,
        "char_count": non_whitespace_char_count(unit.raw_text),
        "raw_char_count": raw_char_count(unit.raw_text),
        "highlight_char_count": highlight_char_count(unit.raw_text),
        "word_count": deterministic_word_count(unit.normalized_text),
        "normalization_trace": unit.normalization_trace,
    }


def _split_identity(
    *,
    book_id: str,
    unit: MaterialUnit | None,
    material_variant: str,
) -> tuple[str, str | None]:
    if unit is None or not unit.normalized_text:
        split_group_id = _stable_id(
            SPLIT_GROUP_ID_VERSION,
            "conservative-unresolved-quarantine",
            book_id,
            material_variant,
        )
        return split_group_id, None
    split_group_id = _stable_id(
        SPLIT_GROUP_ID_VERSION,
        "book-plus-reviewed-identity-text",
        book_id,
        NORMALIZATION_VERSION,
        unit.split_identity_normalized_text,
    )
    return split_group_id, make_content_id(
        book_id=book_id,
        text=unit.split_identity_normalized_text,
    )


def _yield_recording_records(
    *,
    header_path: Path,
    spec: RecordingSpec,
    ce1_catalog: dict[tuple[str, str], list[MaterialUnit]],
    ce2_catalog: dict[str, list[MaterialUnit]],
    audio_spans: dict[tuple[str, str], AudioSpan],
    anomaly_flags: dict[str, set[QualityFlag]],
    split_seed: int,
    diagnostics: dict[str, object],
) -> Iterator[ManifestRecord]:
    match = _HEADER_PATTERN.match(header_path.name)
    if match is None:
        diagnostics["unparsed_headers"].append(str(header_path.resolve()))
        return
    entities = match.groupdict()
    subject_id = entities["subject"]
    session_id = entities["session"]
    run_id = entities["run"]
    book_id = normalize_identifier(session_id)
    if book_id == "granettdream":
        book_id = "garnettdream"
    event_path = header_path.with_name(
        header_path.name.replace("_eeg.vhdr", "_events.tsv")
    )
    event_result = read_event_pairs(event_path)
    if not event_result.pairs:
        diagnostics["recordings_without_legal_pairs"].append(str(header_path.resolve()))
        return
    base_flags = set(anomaly_flags.get(str(header_path.resolve()), set()))
    if event_result.orphan_starts or event_result.orphan_ends:
        base_flags.add(QualityFlag.ORPHAN_ROW_EVENT_IN_RECORDING)

    pace_sec = 0.35 if spec.dataset_version == "ChineseEEG1" else 0.25
    material_variant = "canonical"
    material_evidence = ""
    pairs_for_mapping = list(event_result.pairs)
    prefix_unresolved = 0
    if spec.dataset_version == "ChineseEEG1":
        units = ce1_catalog.get((book_id, str(int(run_id))), [])
        material_variant = "official-segmented-run"
        material_evidence = "ChineseEEG1 reviewed run workbook sequence"
        if event_result.first_chapter_sample is not None:
            prefix_unresolved = sum(
                pair.start_sample < event_result.first_chapter_sample
                for pair in pairs_for_mapping
            )
        mapped_pairs = pairs_for_mapping[prefix_unresolved:]
        mapped_matches = align_event_pairs(
            mapped_pairs,
            units,
            sampling_rate=spec.sampling_rate,
            pace_sec=pace_sec,
            base_evidence=material_evidence,
        )
        matches = [
            UnitMatch(
                None,
                "unresolved",
                None,
                f"{EVENT_ALIGNMENT_VERSION}:before_first_chapter_event",
            )
            for _ in range(prefix_unresolved)
        ] + mapped_matches
    else:
        units, material_variant, material_evidence = select_ce2_run_units(
            book_id=book_id,
            run_id=run_id,
            subject_id=subject_id,
            paradigm=spec.paradigm,
            catalog=ce2_catalog,
        )
        matches = align_event_pairs(
            pairs_for_mapping,
            units,
            sampling_rate=spec.sampling_rate,
            pace_sec=pace_sec,
            base_evidence=material_evidence,
        )

    if len(pairs_for_mapping) != len(units) + prefix_unresolved:
        base_flags.add(QualityFlag.EVENT_TEXT_COUNT_MISMATCH)
    if len(matches) != len(pairs_for_mapping):
        raise RuntimeError("Event alignment did not return one result per legal pair")

    eeg_file, header_resolved = _brainvision_references(header_path)
    speaker_id = None
    if spec.paradigm == "passive_listening":
        speaker_id = "f1" if subject_id in {"01", "02", "03", "04"} else "m1"
    elif spec.paradigm == "reading_aloud":
        speaker_id = subject_id

    for stimulus_position, (pair, unit_match) in enumerate(
        zip(pairs_for_mapping, matches)
    ):
        unit = unit_match.unit
        # Chapter markers provide chapter context but are not stimulus-row trials.
        if unit is not None and unit.is_chapter_marker:
            diagnostics["excluded_chapter_marker_pairs"] += 1
            continue

        flags = set(base_flags)
        eeg_duration_sec = (
            pair.end_sample - pair.start_sample
        ) / spec.sampling_rate
        if eeg_duration_sec < 0.1:
            flags.add(QualityFlag.IMPLAUSIBLE_EEG_TRIAL_DURATION)
        if prefix_unresolved and stimulus_position < prefix_unresolved:
            flags.add(QualityFlag.PRECHAPTER_TRIAL_UNRESOLVED)
        if unit is None:
            flags.update(
                {
                    QualityFlag.MISSING_TEXT,
                    QualityFlag.UNRESOLVED_TEXT_ALIGNMENT,
                }
            )
            if (
                spec.dataset_version == "ChineseEEG2"
                and book_id == "garnettdream"
            ):
                flags.add(QualityFlag.MATERIAL_VARIANT_UNCERTAIN)
        if unit_match.status == "fuzzy":
            flags.add(QualityFlag.EVENT_TEXT_COUNT_MISMATCH)
        if (
            unit is not None
            and unit.global_alignment_status == "unresolved"
            and spec.dataset_version == "ChineseEEG1"
        ):
            flags.add(QualityFlag.UNRESOLVED_TEXT_ALIGNMENT)

        audio_span = None
        audio_alignment_method = "not_applicable_no_audio"
        audio_alignment_evidence = ""
        if spec.paradigm == "reading_aloud":
            flags.add(QualityFlag.RA_AUDIO_BOUNDARY_UNAVAILABLE)
            audio_alignment_method = "null_no_forced_alignment_or_asr_evidence"
            audio_alignment_evidence = (
                "ROWS/ROWE are visual-screen boundaries and are not spoken timing"
            )
        elif spec.paradigm == "passive_listening":
            audio_alignment_method = "null_unverified"
            if book_id == "littleprince" and unit is not None and speaker_id is not None:
                audio_span = audio_spans.get((speaker_id, unit.global_text_id))
            if audio_span is None:
                flags.add(QualityFlag.PL_AUDIO_MAPPING_UNVERIFIED)
                audio_alignment_evidence = (
                    "No row-level audio boundary passed file, speaker/material, "
                    "event-sequence and duration-signature validation"
                )
            else:
                audio_alignment_method = AUDIO_ALIGNMENT_VERSION
                audio_alignment_evidence = audio_span.evidence

        split_group_id, content_id = _split_identity(
            book_id=book_id,
            unit=unit,
            material_variant=material_variant,
        )
        split = assign_split(split_group_id, seed=split_seed)
        sentence_id = (
            unit.sentence_id
            if unit is not None
            else _stable_id(
                SENTENCE_ID_VERSION,
                "unresolved-event-position",
                spec.dataset_version,
                book_id,
                material_variant,
                run_id,
                pair.pair_index,
            )
        )
        global_text_id = unit.global_text_id if unit is not None else None
        record_id = _stable_id(
            RECORD_ID_VERSION,
            spec.dataset_version,
            spec.paradigm,
            subject_id,
            session_id,
            run_id,
            str(header_resolved),
            pair.start_sample,
            pair.end_sample,
        )
        block_id = _stable_id(
            BLOCK_ID_VERSION,
            sentence_id,
            split_group_id,
        )
        text_fields = _manifest_text_fields(unit)
        record = ManifestRecord(
            dataset_version=spec.dataset_version,
            paradigm=spec.paradigm,
            subject_id=subject_id,
            session_id=session_id,
            book_id=book_id,
            chapter_id=unit.chapter_id if unit is not None else None,
            sentence_id=sentence_id,
            global_text_id=global_text_id,
            raw_text=text_fields["raw_text"],
            normalized_text=text_fields["normalized_text"],
            text_hash=text_fields["text_hash"],
            raw_text_hash=text_fields["raw_text_hash"],
            normalized_text_hash=text_fields["normalized_text_hash"],
            text_hash_basis=f"normalized_text/{TEXT_HASH_ALGORITHM}",
            char_count=text_fields["char_count"],
            raw_char_count=text_fields["raw_char_count"],
            highlight_char_count=text_fields["highlight_char_count"],
            char_count_method=CHAR_COUNT_METHOD,
            word_count=text_fields["word_count"],
            word_count_method=jieba_word_count_method(),
            eeg_file=str(eeg_file),
            eeg_start_sample=pair.start_sample,
            eeg_end_sample=pair.end_sample,
            eeg_end_sample_semantics=EEG_END_SAMPLE_SEMANTICS,
            eeg_duration_sec=eeg_duration_sec,
            eeg_sampling_rate=spec.sampling_rate,
            audio_file=audio_span.audio_file if audio_span else None,
            audio_start_sec=audio_span.start_sec if audio_span else None,
            audio_end_sec=audio_span.end_sec if audio_span else None,
            audio_duration_sec=(
                audio_span.end_sec - audio_span.start_sec if audio_span else None
            ),
            event_source=str(event_path.resolve()),
            alignment_source=f"{unit_match.evidence};{material_evidence}",
            quality_flag=encode_quality_flags(flags),
            split_group_id=split_group_id,
            split=split,
            record_id=record_id,
            run_id=run_id,
            block_id=block_id,
            content_id=content_id,
            material_variant=material_variant,
            speaker_id=speaker_id,
            stimulus_position=stimulus_position,
            source_excel_file=unit.source_excel_file if unit is not None else None,
            source_excel_row=unit.source_excel_row if unit is not None else None,
            normalization_version=NORMALIZATION_VERSION,
            normalization_trace=text_fields["normalization_trace"],
            matching_alias=None,
            matching_alias_method="none_no_traditional_simplified_conversion",
            text_alignment_status=unit_match.status,
            text_alignment_score=unit_match.score,
            global_text_alignment_status=(
                unit.global_alignment_status if unit is not None else "unresolved"
            ),
            global_text_alignment_score=(
                unit.global_alignment_score if unit is not None else None
            ),
            audio_alignment_method=audio_alignment_method,
            audio_alignment_evidence=audio_alignment_evidence,
            event_pair_index=pair.pair_index,
            preceding_chapter_event=pair.preceding_chapter_event,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            split_seed=split_seed,
        )
        record.validate()
        yield record


def _recording_specs(paths: ManifestPaths) -> tuple[RecordingSpec, ...]:
    return (
        RecordingSpec(
            dataset_version="ChineseEEG1",
            paradigm="silent_reading",
            sampling_rate=256.0,
            eeg_root=paths.chineseeeg1_eeg_root,
        ),
        RecordingSpec(
            dataset_version="ChineseEEG2",
            paradigm="passive_listening",
            sampling_rate=250.0,
            eeg_root=paths.chineseeeg2_pl_root,
        ),
        RecordingSpec(
            dataset_version="ChineseEEG2",
            paradigm="reading_aloud",
            sampling_rate=250.0,
            eeg_root=paths.chineseeeg2_ra_root,
        ),
    )


def _sorted_headers(root: Path) -> list[Path]:
    def key(path: Path) -> tuple[str, str, tuple[int, str], str]:
        match = _HEADER_PATTERN.match(path.name)
        if match is None:
            return ("", "", (10**9, path.name), str(path))
        entities = match.groupdict()
        return (
            entities["subject"].casefold(),
            entities["session"].casefold(),
            _natural_run_key(entities["run"]),
            str(path).casefold(),
        )

    return sorted(root.rglob("*_eeg.vhdr"), key=key)


def _write_rules(path: Path) -> None:
    rules = {
        "normalization_version": NORMALIZATION_VERSION,
        "hash_algorithm": TEXT_HASH_ALGORITHM,
        "text_hash_basis": "normalized_text",
        "raw_text_policy": "preserve exact source-cell string; never silently edit",
        "simplified_traditional_policy": (
            "no conversion; a future matching alias must be separate and versioned"
        ),
        "rules_in_order": [
            "newline_style_to_lf",
            "remove explicitly enumerated invisible format characters",
            "Unicode NFC (not NFKC)",
            "fullwidth ASCII and ideographic space to ASCII equivalents",
            "explicit quote/book-title/presentation-form punctuation variants",
            "remove linebreaks",
            "remove remaining Unicode whitespace",
        ],
        "preserved_distinctions": [
            "simplified versus traditional Chinese",
            "digits",
            "English letter case",
            "repeated punctuation",
            "lexical characters",
        ],
        "char_count": CHAR_COUNT_METHOD,
        "raw_char_count": RAW_CHAR_COUNT_METHOD,
        "highlight_char_count": (
            "Unicode codepoints excluding categories P, Z and C; corresponds to "
            "the fixed visual highlight clock, not word count"
        ),
        "word_count_method": jieba_word_count_method(),
        "alignment_version": ALIGNMENT_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_trial_manifest(
    *,
    paths: ManifestPaths,
    parquet_path: str | Path,
    csv_path: str | Path,
    normalization_rules_path: str | Path,
    override_path: str | Path,
    diagnostics_path: str | Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    batch_size: int = 4096,
) -> dict[str, object]:
    """Build Parquet and CSV in one deterministic pass."""

    parquet_output = Path(parquet_path)
    csv_output = Path(csv_path)
    diagnostics_output = Path(diagnostics_path)
    for output in (parquet_output, csv_output, diagnostics_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    _write_rules(Path(normalization_rules_path))
    ensure_override_file(override_path)

    ce2_catalog = load_ce2_catalog(paths.chineseeeg2_material_root)
    ce1_catalog, cross_dataset_status = load_ce1_catalog(
        paths.chineseeeg1_novel_root,
        ce2_catalog,
        override_path=override_path,
    )
    audio_spans, audio_diagnostics = load_validated_littleprince_audio_spans(
        paths.chineseeeg2_audio_root,
        ce2_catalog,
    )
    audit = json.loads(paths.audit_json.read_text(encoding="utf-8"))
    anomaly_maps = {
        "ChineseEEG1": _recording_anomaly_flags(audit, "chineseeeg1"),
        "passive_listening": _recording_anomaly_flags(audit, "chineseeeg2_pl"),
        "reading_aloud": _recording_anomaly_flags(audit, "chineseeeg2_ra"),
    }
    diagnostics: dict[str, object] = {
        "builder_version": MANIFEST_SCHEMA_VERSION,
        "split_seed": split_seed,
        "resolved_paths": {
            field: str(value.resolve()) for field, value in asdict(paths).items()
        },
        "cross_dataset_text_alignment": {
            book: dict(sorted(counts.items()))
            for book, counts in cross_dataset_status.items()
        },
        "pl_littleprince_audio_validation": audio_diagnostics,
        "excluded_chapter_marker_pairs": 0,
        "recordings_without_legal_pairs": [],
        "unparsed_headers": [],
        "row_count": 0,
        "paradigm_counts": {},
        "quality_flag_counts": {},
    }
    schema = manifest_arrow_schema()
    columns = schema.names
    parquet_writer = pq.ParquetWriter(
        parquet_output,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    row_count = 0
    paradigm_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    seen_record_ids: set[str] = set()
    split_by_group: dict[str, str] = {}
    batch: list[dict[str, object]] = []
    try:
        with csv_output.open("w", encoding="utf-8", newline="") as csv_handle:
            csv_writer = csv.DictWriter(
                csv_handle,
                fieldnames=columns,
                extrasaction="raise",
                lineterminator="\n",
            )
            csv_writer.writeheader()
            for spec in _recording_specs(paths):
                anomaly_key = (
                    "ChineseEEG1"
                    if spec.dataset_version == "ChineseEEG1"
                    else spec.paradigm
                )
                for header in _sorted_headers(spec.eeg_root):
                    for record in _yield_recording_records(
                        header_path=header,
                        spec=spec,
                        ce1_catalog=ce1_catalog,
                        ce2_catalog=ce2_catalog,
                        audio_spans=audio_spans,
                        anomaly_flags=anomaly_maps[anomaly_key],
                        split_seed=split_seed,
                        diagnostics=diagnostics,
                    ):
                        if record.record_id in seen_record_ids:
                            raise ValueError(f"Duplicate record_id: {record.record_id}")
                        seen_record_ids.add(record.record_id)
                        previous_split = split_by_group.setdefault(
                            record.split_group_id, record.split
                        )
                        if previous_split != record.split:
                            raise ValueError(
                                f"Split leakage for {record.split_group_id}"
                            )
                        row = record.to_dict()
                        csv_writer.writerow(row)
                        batch.append(row)
                        row_count += 1
                        paradigm_counts[record.paradigm] += 1
                        quality_counts[record.quality_flag] += 1
                        if len(batch) >= batch_size:
                            parquet_writer.write_table(
                                pa.Table.from_pylist(batch, schema=schema)
                            )
                            batch.clear()
            if batch:
                parquet_writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
    finally:
        parquet_writer.close()

    diagnostics["row_count"] = row_count
    diagnostics["paradigm_counts"] = dict(sorted(paradigm_counts.items()))
    diagnostics["quality_flag_counts"] = dict(sorted(quality_counts.items()))
    diagnostics["split_group_count"] = len(split_by_group)
    diagnostics_output.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return diagnostics
