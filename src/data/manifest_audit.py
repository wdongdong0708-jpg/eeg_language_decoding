"""Scientific audit for the complete stimulus-row trial manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio

from data.manifest import manifest_arrow_schema
from data.splitting import assign_split

_CONSISTENCY_COLUMNS = (
    "record_id",
    "sentence_id",
    "global_text_id",
    "normalized_text_hash",
    "split_group_id",
    "split",
    "eeg_start_sample",
    "eeg_end_sample",
)


def _count_rows(
    rows: Iterable[dict[str, object]],
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    counts: Counter[tuple[object, ...]] = Counter(
        tuple(row[field] for field in fields) for row in rows
    )
    return [
        {**dict(zip(fields, values)), "trial_count": count}
        for values, count in sorted(
            counts.items(),
            key=lambda item: tuple(
                "" if value is None else str(value) for value in item[0]
            ),
        )
    ]


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    return {
        "count": len(values),
        "min": ordered[0],
        "p05": percentile(0.05),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _digest_update(digest: object, values: Iterable[object]) -> None:
    for value in values:
        encoded = "" if value is None else str(value)
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\x1f")
    digest.update(b"\n")


def _parquet_key_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        _digest_update(digest, (row[column] for column in _CONSISTENCY_COLUMNS))
    return digest.hexdigest()


def _csv_consistency(path: Path) -> tuple[int, str, list[str]]:
    digest = hashlib.sha256()
    count = 0
    record_ids: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count += 1
            record_ids.append(row["record_id"])
            _digest_update(digest, (row[column] for column in _CONSISTENCY_COLUMNS))
    return count, digest.hexdigest(), record_ids


def _near_duplicate_audit(
    rows: list[dict[str, object]],
    *,
    minimum_score: float = 96.0,
) -> dict[str, object]:
    unique: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        text = row["normalized_text"]
        if not text:
            continue
        unique.setdefault(
            (str(row["book_id"]), str(text)),
            {
                "book_id": row["book_id"],
                "normalized_text": text,
                "normalized_text_hash": row["normalized_text_hash"],
            },
        )
    buckets: dict[tuple[str, int, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in unique.values():
        text = str(item["normalized_text"])
        buckets[(str(item["book_id"]), len(text), text[:1], text[-1:])].append(item)

    pairs: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for (book_id, length, first, last), source_items in buckets.items():
        candidates: list[dict[str, object]] = []
        for delta in (-2, -1, 0, 1, 2):
            candidates.extend(
                buckets.get((book_id, length + delta, first, last), [])
            )
        for source in source_items:
            source_text = str(source["normalized_text"])
            for target in candidates:
                target_text = str(target["normalized_text"])
                if source_text == target_text:
                    continue
                identity = tuple(sorted((source_text, target_text)))
                if identity in seen:
                    continue
                seen.add(identity)
                score = float(ratio(source_text, target_text))
                if score >= minimum_score:
                    pairs.append(
                        {
                            "book_id": book_id,
                            "score": score,
                            "text_a": source_text,
                            "text_b": target_text,
                        }
                    )
    pairs.sort(key=lambda item: (-float(item["score"]), str(item["text_a"])))
    return {
        "method": (
            "rapidfuzz ratio >=96; candidate lengths within 2 codepoints; "
            "diagnostic only, never used to merge split groups"
        ),
        "pair_count": len(pairs),
        "preview": pairs[:100],
    }


def _source_event_anomalies(data_audit: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for dataset_key, details in data_audit["eeg"].items():
        recordings = details["recordings"]
        result[dataset_key] = {
            "recording_count": len(recordings),
            "missing_events_recording_count": sum(
                not record["events_tsv_exists"] for record in recordings
            ),
            "orphan_rows_count": sum(
                int(record["orphan_row_starts"]) + int(record["orphan_row_ends"])
                for record in recordings
            ),
            "orphan_row_starts": sum(
                int(record["orphan_row_starts"]) for record in recordings
            ),
            "orphan_row_ends": sum(
                int(record["orphan_row_ends"]) for record in recordings
            ),
            "events_vmrk_count_mismatch_recording_count": sum(
                record["event_count"] != record["annotation_count_vmrk"]
                for record in recordings
            ),
            "broken_brainvision_reference_recording_count": sum(
                not record["header_data_reference_exists"]
                or not record["header_marker_reference_exists"]
                for record in recordings
            ),
            "missing_events_recordings": [
                record["path"] for record in recordings if not record["events_tsv_exists"]
            ],
        }
    return result


def audit_manifest(
    *,
    parquet_path: str | Path,
    csv_path: str | Path,
    data_audit_path: str | Path,
    build_diagnostics_path: str | Path,
) -> dict[str, object]:
    parquet_file = Path(parquet_path)
    csv_file = Path(csv_path)
    table = pq.read_table(parquet_file)
    rows = table.to_pylist()
    data_audit = json.loads(Path(data_audit_path).read_text(encoding="utf-8"))
    diagnostics = json.loads(
        Path(build_diagnostics_path).read_text(encoding="utf-8")
    )

    quality_flag_counts: Counter[str] = Counter()
    for row in rows:
        quality_flag_counts.update(str(row["quality_flag"]).split("|"))
    normalization_rule_trial_hits: Counter[str] = Counter()
    normalization_rule_character_hits: Counter[str] = Counter()
    for row in rows:
        for change in json.loads(str(row["normalization_trace"])):
            normalization_rule_trial_hits[change["rule"]] += 1
            normalization_rule_character_hits[change["rule"]] += int(change["count"])

    split_by_group: dict[str, set[str]] = defaultdict(set)
    expected_split_mismatches: list[dict[str, object]] = []
    group_trial_counts: Counter[str] = Counter()
    group_paradigms: dict[str, set[str]] = defaultdict(set)
    group_text_hashes: dict[str, set[str]] = defaultdict(set)
    global_hashes: dict[str, set[str]] = defaultdict(set)
    global_exact_hashes: dict[str, set[str]] = defaultdict(set)
    global_statuses: dict[str, set[str]] = defaultdict(set)
    group_global_statuses: dict[str, set[str]] = defaultdict(set)
    sentence_ids_by_text_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = str(row["split_group_id"])
        split = str(row["split"])
        split_by_group[group_id].add(split)
        group_trial_counts[group_id] += 1
        group_paradigms[group_id].add(str(row["paradigm"]))
        if row["normalized_text_hash"]:
            group_text_hashes[group_id].add(str(row["normalized_text_hash"]))
            group_global_statuses[group_id].add(
                str(row["global_text_alignment_status"])
            )
            sentence_ids_by_text_hash[str(row["normalized_text_hash"])].add(
                str(row["sentence_id"])
            )
        if row["global_text_id"] and row["normalized_text_hash"]:
            global_id = str(row["global_text_id"])
            normalized_hash = str(row["normalized_text_hash"])
            global_hashes[global_id].add(normalized_hash)
            status = str(row["global_text_alignment_status"])
            global_statuses[global_id].add(status)
            if status in {"exact", "normalized", "manual"}:
                global_exact_hashes[global_id].add(normalized_hash)
        expected = assign_split(group_id, seed=int(row["split_seed"]))
        if expected != split and len(expected_split_mismatches) < 100:
            expected_split_mismatches.append(
                {
                    "record_id": row["record_id"],
                    "split_group_id": group_id,
                    "observed": split,
                    "expected": expected,
                }
            )

    leakage = {
        group: sorted(splits)
        for group, splits in split_by_group.items()
        if len(splits) > 1
    }
    global_conflicts = {
        global_id: sorted(hashes)
        for global_id, hashes in global_exact_hashes.items()
        if len(hashes) > 1
    }
    accepted_global_variants = {
        global_id: {
            "normalized_text_hashes": sorted(hashes),
            "alignment_statuses": sorted(global_statuses[global_id]),
        }
        for global_id, hashes in global_hashes.items()
        if len(hashes) > 1 and global_id not in global_conflicts
    }
    split_group_text_conflicts = {
        group_id: sorted(hashes)
        for group_id, hashes in group_text_hashes.items()
        if len(hashes) > 1
        and "fuzzy" not in group_global_statuses[group_id]
    }

    csv_row_count, csv_digest, csv_record_ids = _csv_consistency(csv_file)
    parquet_digest = _parquet_key_digest(rows)
    parquet_record_ids = [str(row["record_id"]) for row in rows]
    schema = manifest_arrow_schema()
    schema_equal = table.schema.equals(schema, check_metadata=True)

    exact_duplicates = [
        {
            "normalized_text_hash": text_hash,
            "distinct_sentence_id_count": len(sentence_ids),
            "sentence_id_preview": sorted(sentence_ids)[:20],
        }
        for text_hash, sentence_ids in sentence_ids_by_text_hash.items()
        if len(sentence_ids) > 1
    ]
    exact_duplicates.sort(
        key=lambda item: (-int(item["distinct_sentence_id_count"]), item["normalized_text_hash"])
    )
    coverage_counts = Counter(
        "+".join(sorted(paradigms)) for paradigms in group_paradigms.values()
    )

    unresolved_text_groups = _count_rows(
        [row for row in rows if row["raw_text"] is None],
        ("dataset_version", "paradigm", "subject_id", "book_id", "run_id", "quality_flag"),
    )
    unresolved_audio_groups = _count_rows(
        [
            row
            for row in rows
            if row["paradigm"] in {"passive_listening", "reading_aloud"}
            and row["audio_start_sec"] is None
        ],
        ("paradigm", "subject_id", "book_id", "run_id", "audio_alignment_method"),
    )

    split_trial_counts = Counter(str(row["split"]) for row in rows)
    split_group_counts = Counter(
        next(iter(splits)) for splits in split_by_group.values()
    )
    audit = {
        "audit_version": "manifest-audit-v1",
        "manifest": {
            "parquet": str(parquet_file.resolve()),
            "csv": str(csv_file.resolve()),
            "row_count": len(rows),
            "schema_equal_authoritative": schema_equal,
            "schema": str(table.schema),
            "column_count": table.num_columns,
            "record_id_unique": len(set(parquet_record_ids)) == len(rows),
            "deterministic_order_definition": (
                "dataset order ChineseEEG1/PL/RA, subject, session, numeric run, "
                "EEG event sample"
            ),
        },
        "trial_counts": {
            "by_dataset_paradigm": _count_rows(
                rows, ("dataset_version", "paradigm")
            ),
            "by_dataset_paradigm_subject_book_chapter": _count_rows(
                rows,
                (
                    "dataset_version",
                    "paradigm",
                    "subject_id",
                    "book_id",
                    "chapter_id",
                ),
            ),
        },
        "missingness": {
            "missing_text_trial_count": sum(row["raw_text"] is None for row in rows),
            "missing_event_recording_count": sum(
                value["missing_events_recording_count"]
                for value in _source_event_anomalies(data_audit).values()
            ),
            "missing_audio_boundary_trial_count_all_paradigms": sum(
                row["audio_start_sec"] is None for row in rows
            ),
            "missing_audio_boundary_passive_listening": sum(
                row["paradigm"] == "passive_listening"
                and row["audio_start_sec"] is None
                for row in rows
            ),
            "missing_audio_boundary_reading_aloud_expected": sum(
                row["paradigm"] == "reading_aloud"
                and row["audio_start_sec"] is None
                for row in rows
            ),
        },
        "source_event_anomalies": _source_event_anomalies(data_audit),
        "durations_sec": {
            "eeg": _distribution([float(row["eeg_duration_sec"]) for row in rows]),
            "audio_non_null": _distribution(
                [
                    float(row["audio_duration_sec"])
                    for row in rows
                    if row["audio_duration_sec"] is not None
                ]
            ),
            "eeg_by_paradigm": {
                paradigm: _distribution(
                    [
                        float(row["eeg_duration_sec"])
                        for row in rows
                        if row["paradigm"] == paradigm
                    ]
                )
                for paradigm in sorted({str(row["paradigm"]) for row in rows})
            },
        },
        "normalization": {
            "rule_trial_hits": dict(sorted(normalization_rule_trial_hits.items())),
            "rule_character_hits": dict(
                sorted(normalization_rule_character_hits.items())
            ),
            "version_counts": dict(
                sorted(Counter(str(row["normalization_version"]) for row in rows).items())
            ),
        },
        "text_alignment": {
            "event_to_source_status_counts": dict(
                sorted(Counter(str(row["text_alignment_status"]) for row in rows).items())
            ),
            "global_status_counts": dict(
                sorted(
                    Counter(
                        str(row["global_text_alignment_status"]) for row in rows
                    ).items()
                )
            ),
            "global_text_id_conflict_count": len(global_conflicts),
            "global_text_id_conflicts": global_conflicts,
            "accepted_fuzzy_global_variant_count": len(accepted_global_variants),
            "accepted_fuzzy_global_variants": accepted_global_variants,
            "split_group_text_conflict_count": len(split_group_text_conflicts),
            "split_group_text_conflicts": split_group_text_conflicts,
        },
        "quality_flags": dict(sorted(quality_flag_counts.items())),
        "splits": {
            "seed_values": sorted({int(row["split_seed"]) for row in rows}),
            "trial_counts": dict(sorted(split_trial_counts.items())),
            "content_group_counts": dict(sorted(split_group_counts.items())),
            "split_group_count": len(split_by_group),
            "cross_split_leakage_count": len(leakage),
            "cross_split_leakage": leakage,
            "hash_assignment_mismatch_count": len(expected_split_mismatches),
            "hash_assignment_mismatch_preview": expected_split_mismatches,
        },
        "duplicates": {
            "exact_normalized_text_groups_with_multiple_sentence_ids": len(
                exact_duplicates
            ),
            "exact_duplicate_preview": exact_duplicates[:100],
            "near_duplicates": _near_duplicate_audit(rows),
        },
        "cross_paradigm_coverage": {
            "split_group_paradigm_combination_counts": dict(
                sorted(coverage_counts.items())
            ),
            "all_three_paradigms_group_count": sum(
                paradigms
                == {"silent_reading", "passive_listening", "reading_aloud"}
                for paradigms in group_paradigms.values()
            ),
        },
        "unresolved": {
            "text_grouped_list": unresolved_text_groups,
            "audio_grouped_list": unresolved_audio_groups,
            "recordings_without_legal_pairs": diagnostics[
                "recordings_without_legal_pairs"
            ],
        },
        "chapter_markers": {
            "excluded_pair_count": diagnostics["excluded_chapter_marker_pairs"],
            "policy": "chapter markers assign context but are not emitted as trials",
        },
        "parquet_csv_consistency": {
            "parquet_row_count": len(rows),
            "csv_row_count": csv_row_count,
            "row_count_equal": len(rows) == csv_row_count,
            "record_id_order_equal": parquet_record_ids == csv_record_ids,
            "critical_field_sha256_parquet": parquet_digest,
            "critical_field_sha256_csv": csv_digest,
            "critical_field_digest_equal": parquet_digest == csv_digest,
            "critical_fields": list(_CONSISTENCY_COLUMNS),
        },
        "build_diagnostics": diagnostics,
    }
    return audit


def render_manifest_audit_markdown(audit: dict[str, object]) -> str:
    manifest = audit["manifest"]
    missing = audit["missingness"]
    split = audit["splits"]
    consistency = audit["parquet_csv_consistency"]
    text_alignment = audit["text_alignment"]
    lines = [
        "# Trial Manifest Audit",
        "",
        "This report audits complete stimulus-row trials. `sentence_id` denotes an "
        "experimental stimulus row/short segment; it is not asserted to be a "
        "linguistic sentence or a word-aligned unit.",
        "",
        "## Contract",
        "",
        "- Only legal `ROWS` followed by `ROWE` pairs are emitted.",
        "- EEG intervals use half-open `[eeg_start_sample, eeg_end_sample)` bounds; "
        "`eeg_end_sample` is the exclusive ROWE event-onset sample.",
        "- Chapter-marker pairs are excluded from trials and retained only as context.",
        "- Raw workbook text is preserved; normalization is separate and traced.",
        "- RA screen events are never used as spoken-audio boundaries.",
        "- Split assignment is SHA-256 over versioned content groups with seed "
        f"`{split['seed_values']}`.",
        "",
        "## Manifest",
        "",
        f"- Rows: {manifest['row_count']:,}",
        f"- Columns: {manifest['column_count']}",
        f"- Unique primary keys: `{manifest['record_id_unique']}`",
        f"- Authoritative schema match: `{manifest['schema_equal_authoritative']}`",
        f"- Excluded chapter-marker event pairs: "
        f"{audit['chapter_markers']['excluded_pair_count']:,}",
        "",
        "### Dataset/paradigm counts",
        "",
        "| dataset | paradigm | trials |",
        "|---|---|---:|",
    ]
    for row in audit["trial_counts"]["by_dataset_paradigm"]:
        lines.append(
            f"| {row['dataset_version']} | {row['paradigm']} | "
            f"{row['trial_count']:,} |"
        )
    lines.extend(
        [
            "",
            "The full dataset/paradigm/subject/book/chapter table is available in "
            "`reports/manifest_audit.json`.",
            "",
            "## Missingness and event integrity",
            "",
            f"- Trials with unresolved local text: "
            f"{missing['missing_text_trial_count']:,}",
            f"- Recordings missing event TSV: "
            f"{missing['missing_event_recording_count']:,}",
            f"- PL trials without validated audio bounds: "
            f"{missing['missing_audio_boundary_passive_listening']:,}",
            f"- RA trials with intentionally null audio bounds: "
            f"{missing['missing_audio_boundary_reading_aloud_expected']:,}",
            "",
            "### Source event anomalies",
            "",
            "| source | missing events recordings | orphan ROW markers | "
            "events/vmrk mismatch recordings | broken references |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, details in audit["source_event_anomalies"].items():
        lines.append(
            f"| {dataset} | {details['missing_events_recording_count']} | "
            f"{details['orphan_rows_count']} | "
            f"{details['events_vmrk_count_mismatch_recording_count']} | "
            f"{details['broken_brainvision_reference_recording_count']} |"
        )
    lines.extend(
        [
            "",
            "## Alignment",
            "",
            f"- Event-to-source statuses: "
            f"`{json.dumps(text_alignment['event_to_source_status_counts'], ensure_ascii=False)}`",
            f"- Cross-dataset/global statuses: "
            f"`{json.dumps(text_alignment['global_status_counts'], ensure_ascii=False)}`",
            f"- `global_text_id` conflicts: "
            f"{text_alignment['global_text_id_conflict_count']}",
            "",
            "ChineseEEG1-to-ChineseEEG2 sharing uses monotonic exact anchors and "
            "reciprocal fuzzy matches inside anchor gaps. It never joins by row "
            "number across datasets. Unaccepted matches retain dataset-local IDs.",
            "",
            "## Quality flags",
            "",
            "| flag | trials |",
            "|---|---:|",
        ]
    )
    for flag, count in audit["quality_flags"].items():
        lines.append(f"| {flag} | {count:,} |")
    lines.extend(
        [
            "",
            "## Deterministic splits and leakage",
            "",
            f"- Content groups: {split['split_group_count']:,}",
            f"- Group counts: "
            f"`{json.dumps(split['content_group_counts'], ensure_ascii=False)}`",
            f"- Trial counts: `{json.dumps(split['trial_counts'], ensure_ascii=False)}`",
            f"- Same `split_group_id` crossing splits: "
            f"**{split['cross_split_leakage_count']}**",
            f"- Stored split versus fixed-hash mismatch: "
            f"**{split['hash_assignment_mismatch_count']}**",
            "",
            "## Duration and shortcut-relevant fields",
            "",
            f"- EEG seconds: `{json.dumps(audit['durations_sec']['eeg'], ensure_ascii=False)}`",
            f"- Validated audio seconds: "
            f"`{json.dumps(audit['durations_sec']['audio_non_null'], ensure_ascii=False)}`",
            "",
            "`char_count`, `raw_char_count`, `highlight_char_count`, EEG/audio "
            "duration and padding-relevant boundaries are explicit manifest fields "
            "for later shortcut baselines and length-matched candidate pools.",
            "",
            "## Normalization and duplicate diagnostics",
            "",
            f"- Rule trial hits: "
            f"`{json.dumps(audit['normalization']['rule_trial_hits'], ensure_ascii=False)}`",
            f"- Exact repeated normalized-text groups: "
            f"{audit['duplicates']['exact_normalized_text_groups_with_multiple_sentence_ids']:,}",
            f"- Near-duplicate diagnostic pairs: "
            f"{audit['duplicates']['near_duplicates']['pair_count']:,}",
            f"- Split groups covered by all three paradigms: "
            f"{audit['cross_paradigm_coverage']['all_three_paradigms_group_count']:,}",
            "",
            "Near-duplicate detection is audit-only and never merges identities.",
            "",
            "## Parquet/CSV consistency",
            "",
            f"- Row count equal: `{consistency['row_count_equal']}`",
            f"- Primary-key order equal: `{consistency['record_id_order_equal']}`",
            f"- Critical-field digest equal: "
            f"`{consistency['critical_field_digest_equal']}`",
            f"- Digest: `{consistency['critical_field_sha256_parquet']}`",
            "",
            "## Conservative unresolved mappings",
            "",
            f"- Grouped unresolved text entries: "
            f"{len(audit['unresolved']['text_grouped_list'])}",
            f"- Grouped null audio entries: "
            f"{len(audit['unresolved']['audio_grouped_list'])}",
            f"- Recordings with no legal trial pair: "
            f"{len(audit['unresolved']['recordings_without_legal_pairs'])}",
            "",
            "The complete grouped lists and source recording paths are in "
            "`reports/manifest_audit.json`. Null values are intentional wherever "
            "the evidence does not support an alignment.",
            "",
        ]
    )
    return "\n".join(lines)
