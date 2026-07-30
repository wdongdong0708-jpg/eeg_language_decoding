"""Versioned, deterministic protocol-level splits for the ChineseEEG manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow.parquet as pq

from data.splitting import SplitRatios, assign_content_groups, assign_groups_by_quota

PROTOCOL_VERSION = "split-protocol-v1"
SUBJECT_GROUP_VERSION = "subject-group-v1"
DEFAULT_PROTOCOL_SEED = 42
PARTITIONS = ("train", "validation", "test")
_INTERNAL_TO_PROTOCOL = {"train": "train", "valid": "validation", "test": "test"}
_CRITICAL_FIELDS = (
    "record_id",
    "sentence_id",
    "global_text_id",
    "normalized_text_hash",
    "split_group_id",
    "split",
    "eeg_start_sample",
    "eeg_end_sample",
)
_REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            *_CRITICAL_FIELDS,
            "dataset_version",
            "paradigm",
            "subject_id",
            "raw_text",
            "global_text_alignment_status",
            "quality_flag",
            "manifest_schema_version",
            "split_seed",
        )
    )
)
_CROSS_PROTOCOLS = (
    (
        "pl_to_silent_reading_unseen_text",
        ("passive_listening",),
        "silent_reading",
    ),
    (
        "silent_reading_to_pl_unseen_text",
        ("silent_reading",),
        "passive_listening",
    ),
    (
        "pl_silent_reading_to_ra_unseen_text",
        ("passive_listening", "silent_reading"),
        "reading_aloud",
    ),
)


def subject_namespace(row: Mapping[str, object]) -> str:
    """Return a cohort-namespaced subject identity, never a bare subject string."""

    return "::".join(
        (
            str(row["dataset_version"]),
            str(row["paradigm"]),
            str(row["subject_id"]),
        )
    )


def make_subject_group_id(row: Mapping[str, object]) -> str:
    namespace = subject_namespace(row)
    payload = f"{SUBJECT_GROUP_VERSION}\0{namespace}".encode("utf-8")
    return f"{SUBJECT_GROUP_VERSION}-{hashlib.sha256(payload).hexdigest()[:24]}"


def load_protocol_manifest(path: str | Path) -> list[dict[str, object]]:
    """Read only protocol-relevant fields without modifying the manifest."""

    table = pq.read_table(path, columns=list(_REQUIRED_COLUMNS))
    rows = table.to_pylist()
    _validate_rows(rows)
    return rows


def manifest_critical_field_fingerprint(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Match the manifest audit's physical-row-order critical-field digest."""

    digest = hashlib.sha256()
    for row in rows:
        for field in _CRITICAL_FIELDS:
            value = row[field]
            digest.update(("" if value is None else str(value)).encode("utf-8"))
            digest.update(b"\x1f")
        digest.update(b"\n")
    return {
        "algorithm": "sha256",
        "fields": list(_CRITICAL_FIELDS),
        "row_order": "physical manifest row order",
        "value_separator_hex": "1f",
        "row_separator_hex": "0a",
        "sha256": digest.hexdigest(),
    }


def build_text_unseen_protocol(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_path: str = "metadata/all_trials.parquet",
    seed: int = DEFAULT_PROTOCOL_SEED,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, object]:
    """Setting A: unseen content with subjects allowed to overlap."""

    prepared = _prepare_rows(rows)
    content_assignments = _content_assignments(prepared, seed=seed, ratios=ratios)
    selected = {partition: [] for partition in PARTITIONS}
    for row in prepared:
        partition = content_assignments[str(row["split_group_id"])]
        selected[partition].append(row)
    artifact = _base_artifact(
        prepared,
        manifest_path=manifest_path,
        seed=seed,
        ratios=ratios,
        setting="A_text_unseen_subject_seen",
        assumptions=[
            "Partitioning is decided once per split_group_id; trials only inherit it.",
            "Subjects may occur in multiple partitions, and their overlap is reported.",
            "No quality_flag, fuzzy status, missing text, duration, or trial index is used "
            "to assign partitions.",
            "Ratios are approximate at content-group level, not exact at trial level.",
        ],
    )
    _attach_selection(
        artifact,
        selected,
        excluded_by_reason={},
        all_rows=prepared,
        all_content_assignments=content_assignments,
    )
    train_subjects = set(artifact["partitions"]["train"]["subject_group_ids"])
    test_subjects = set(artifact["partitions"]["test"]["subject_group_ids"])
    artifact["subject_overlap"] = {
        "train_test_subject_group_ids": sorted(train_subjects & test_subjects),
        "train_test_subject_overlap_count": len(train_subjects & test_subjects),
        "allowed_by_protocol": True,
    }
    artifact["leakage_checks"]["train_test_subject_overlap_allowed"] = True
    return artifact


def build_subject_text_unseen_protocol(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_path: str = "metadata/all_trials.parquet",
    seed: int = DEFAULT_PROTOCOL_SEED,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, object]:
    """Setting B: strict subject/content diagonal with complete exclusion accounting."""

    prepared = _prepare_rows(rows)
    content_assignments = _content_assignments(prepared, seed=seed, ratios=ratios)
    subject_assignments, cohort_details, registry = _subject_assignments(
        prepared, seed=seed, ratios=ratios
    )
    selected = {partition: [] for partition in PARTITIONS}
    excluded: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prepared:
        subject_partition = subject_assignments[str(row["_subject_group_id"])]
        content_partition = content_assignments[str(row["split_group_id"])]
        if subject_partition == content_partition:
            selected[subject_partition].append(row)
        else:
            reason = (
                f"off_diagonal_subject_{subject_partition}"
                f"__content_{content_partition}"
            )
            excluded[reason].append(row)

    artifact = _base_artifact(
        prepared,
        manifest_path=manifest_path,
        seed=seed,
        ratios=ratios,
        setting="B_subject_and_text_unseen_strict_diagonal",
        assumptions=[
            "Content groups use SHA-256 threshold assignment from split_group_id.",
            "Subject groups are independently hash-ranked within dataset/paradigm "
            "cohorts and assigned integer quotas.",
            "Validation and test subject quotas are kept non-empty for every cohort "
            "with at least three subject groups.",
            "Only diagonal subject-partition × content-partition cells are selected; "
            "all off-diagonal cells are explicitly excluded.",
            "No quality field changes content or subject assignment.",
        ],
    )
    artifact["subject_assignment"] = {
        "algorithm": "cohort-local SHA-256 hash ranking followed by integer quota allocation",
        "cohorts": cohort_details,
        "registry": registry,
        "subject_group_ids": {
            partition: sorted(
                group_id
                for group_id, assignment in subject_assignments.items()
                if assignment == partition
            )
            for partition in PARTITIONS
        },
    }
    _attach_selection(
        artifact,
        selected,
        excluded_by_reason=excluded,
        all_rows=prepared,
        all_content_assignments=content_assignments,
    )
    artifact["leakage_checks"]["train_test_subject_group_intersection_count"] = len(
        set(artifact["partitions"]["train"]["subject_group_ids"])
        & set(artifact["partitions"]["test"]["subject_group_ids"])
    )
    return artifact


def build_cross_paradigm_protocol(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_path: str = "metadata/all_trials.parquet",
    seed: int = DEFAULT_PROTOCOL_SEED,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, object]:
    """Setting C: three zero-shot paradigm-transfer protocols with unseen text."""

    prepared = _prepare_rows(rows)
    content_assignments = _content_assignments(prepared, seed=seed, ratios=ratios)
    strict_group_masks = _strict_cross_paradigm_group_masks(prepared)
    masked_groups = set().union(*strict_group_masks.values())

    artifact = _base_artifact(
        prepared,
        manifest_path=manifest_path,
        seed=seed,
        ratios=ratios,
        setting="C_cross_paradigm_zero_shot_unseen_text",
        assumptions=[
            "Each main protocol is zero-shot with respect to the target EEG paradigm.",
            "Train/validation use source-paradigm EEG only; test uses target-paradigm "
            "EEG only.",
            "Content partitions are global and deterministic from split_group_id.",
            "Groups implicated by fuzzy global alignment, missing text, or uncertain "
            "material variants are conservatively excluded from strict main protocols.",
            "Other quality flags, including implausible duration, do not filter rows.",
            "Unresolved global alignment is retained: unseen-text transfer does not "
            "require paired source/target trials, but this remains a reported risk.",
        ],
    )
    artifact["strict_sensitivity_masks"] = {
        "policy": "whole split_group_id is excluded if any row triggers a mask",
        "masks": {
            reason: {
                "split_group_ids": sorted(group_ids),
                "content_group_count": len(group_ids),
                "trial_count_in_manifest": sum(
                    str(row["split_group_id"]) in group_ids for row in prepared
                ),
            }
            for reason, group_ids in strict_group_masks.items()
        },
        "union_content_group_count": len(masked_groups),
        "union_trial_count_in_manifest": sum(
            str(row["split_group_id"]) in masked_groups for row in prepared
        ),
    }
    content_counts = Counter(content_assignments.values())
    artifact["actual_ratios"] = {
        "all_content_groups": {
            partition: content_counts.get(partition, 0) / len(content_assignments)
            for partition in PARTITIONS
        }
    }
    artifact["content_assignment"] = _content_assignment_payload(
        content_assignments
    )
    protocols: dict[str, object] = {}
    for name, source_paradigms, target_paradigm in _CROSS_PROTOCOLS:
        selected = {partition: [] for partition in PARTITIONS}
        excluded: dict[str, list[dict[str, object]]] = defaultdict(list)
        source_set = set(source_paradigms)
        for row in prepared:
            group_id = str(row["split_group_id"])
            paradigm = str(row["paradigm"])
            partition = content_assignments[group_id]
            mask_reason = _mask_reason(group_id, strict_group_masks)
            if mask_reason is not None:
                excluded[mask_reason].append(row)
            elif paradigm in source_set:
                if partition in {"train", "validation"}:
                    selected[partition].append(row)
                else:
                    excluded["source_paradigm_test_content_excluded"].append(row)
            elif paradigm == target_paradigm:
                if partition == "test":
                    selected["test"].append(row)
                else:
                    excluded["target_paradigm_non_test_content_excluded"].append(row)
            else:
                excluded["paradigm_out_of_protocol_scope"].append(row)

        protocol: dict[str, object] = {
            "name": name,
            "transfer_type": "zero-shot paradigm transfer + unseen text",
            "source_paradigms": list(source_paradigms),
            "target_paradigm": target_paradigm,
        }
        _attach_selection(
            protocol,
            selected,
            excluded_by_reason=excluded,
            all_rows=prepared,
            all_content_assignments=content_assignments,
        )
        train_target_count = sum(
            row["paradigm"] == target_paradigm
            for partition in ("train", "validation")
            for row in selected[partition]
        )
        protocol["leakage_checks"]["target_paradigm_in_train_or_validation_count"] = (
            train_target_count
        )
        protocols[name] = protocol
    artifact["protocols"] = protocols
    artifact["protocol_counts"] = {
        name: protocol["counts"] for name, protocol in protocols.items()
    }
    artifact["leakage_checks"] = {
        name: protocol["leakage_checks"] for name, protocol in protocols.items()
    }
    artifact["supplementary_protocols"] = {}
    return artifact


def build_all_protocols(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_path: str = "metadata/all_trials.parquet",
    seed: int = DEFAULT_PROTOCOL_SEED,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, dict[str, object]]:
    return {
        f"text_unseen_seed{seed}.json": build_text_unseen_protocol(
            rows, manifest_path=manifest_path, seed=seed, ratios=ratios
        ),
        f"subject_text_unseen_seed{seed}.json": build_subject_text_unseen_protocol(
            rows, manifest_path=manifest_path, seed=seed, ratios=ratios
        ),
        f"cross_paradigm_seed{seed}.json": build_cross_paradigm_protocol(
            rows, manifest_path=manifest_path, seed=seed, ratios=ratios
        ),
    }


def write_json_deterministic(path: str | Path, payload: Mapping[str, object]) -> str:
    """Write canonical project JSON and return its SHA-256."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8", newline="\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Protocol splitting requires a non-empty manifest")
    missing_columns = sorted(set(_REQUIRED_COLUMNS) - set(rows[0]))
    if missing_columns:
        raise ValueError(f"Manifest is missing protocol fields: {missing_columns}")
    record_ids = [row["record_id"] for row in rows]
    if any(not value for value in record_ids):
        raise ValueError("record_id cannot be null or empty")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record_id must be unique")
    if any(not row["split_group_id"] for row in rows):
        raise ValueError("split_group_id cannot be null or empty")
    schema_versions = {str(row["manifest_schema_version"]) for row in rows}
    if len(schema_versions) != 1:
        raise ValueError(f"Expected one manifest schema version, got {schema_versions}")


def _prepare_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    _validate_rows(rows)
    prepared: list[dict[str, object]] = []
    namespace_by_group: dict[str, str] = {}
    for source in rows:
        row = dict(source)
        group_id = make_subject_group_id(row)
        namespace = subject_namespace(row)
        previous = namespace_by_group.setdefault(group_id, namespace)
        if previous != namespace:
            raise ValueError(f"subject_group_id hash collision: {group_id}")
        row["_subject_group_id"] = group_id
        row["_subject_namespace"] = namespace
        prepared.append(row)
    return prepared


def _content_assignments(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    ratios: SplitRatios,
) -> dict[str, str]:
    internal = assign_content_groups(
        (str(row["split_group_id"]) for row in rows),
        seed=seed,
        ratios=ratios,
    )
    return {group_id: _INTERNAL_TO_PROTOCOL[value] for group_id, value in internal.items()}


def _subject_assignments(
    rows: Sequence[Mapping[str, object]],
    *,
    seed: int,
    ratios: SplitRatios,
) -> tuple[dict[str, str], list[dict[str, object]], list[dict[str, str]]]:
    cohort_groups: dict[str, set[str]] = defaultdict(set)
    registry_map: dict[str, tuple[str, str]] = {}
    for row in rows:
        cohort = f"{row['dataset_version']}::{row['paradigm']}"
        group_id = str(row["_subject_group_id"])
        cohort_groups[cohort].add(group_id)
        registry_map[group_id] = (cohort, str(row["_subject_namespace"]))

    assignments: dict[str, str] = {}
    cohort_details: list[dict[str, object]] = []
    for cohort in sorted(cohort_groups):
        group_ids = sorted(cohort_groups[cohort])
        internal = assign_groups_by_quota(
            group_ids,
            seed=f"{seed}\0{cohort}",
            ratios=ratios,
            require_validation_and_test=True,
        )
        mapped = {
            group_id: _INTERNAL_TO_PROTOCOL[partition]
            for group_id, partition in internal.items()
        }
        assignments.update(mapped)
        counts = Counter(mapped.values())
        cohort_details.append(
            {
                "cohort": cohort,
                "subject_group_count": len(group_ids),
                "requested_ratios": _ratio_dict(ratios),
                "assigned_counts": {
                    partition: counts.get(partition, 0) for partition in PARTITIONS
                },
                "actual_ratios": {
                    partition: counts.get(partition, 0) / len(group_ids)
                    for partition in PARTITIONS
                },
                "ratio_deviation": {
                    partition: (
                        counts.get(partition, 0) / len(group_ids)
                        - _ratio_dict(ratios)[partition]
                    )
                    for partition in PARTITIONS
                },
            }
        )
    registry = [
        {
            "subject_group_id": group_id,
            "cohort": registry_map[group_id][0],
            "subject_namespace": registry_map[group_id][1],
        }
        for group_id in sorted(registry_map)
    ]
    return assignments, cohort_details, registry


def _base_artifact(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest_path: str,
    seed: int,
    ratios: SplitRatios,
    setting: str,
    assumptions: list[str],
) -> dict[str, object]:
    schema_version = str(rows[0]["manifest_schema_version"])
    embedded_seeds = sorted({int(row["split_seed"]) for row in rows})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_from_manifest_schema_version": schema_version,
        "manifest_path": Path(manifest_path).as_posix(),
        "manifest_row_count": len(rows),
        "manifest_critical_field_fingerprint": manifest_critical_field_fingerprint(rows),
        "manifest_embedded_split": {
            "preserved": True,
            "seed_values": embedded_seeds,
            "note": (
                "Protocol partitions are independent artifacts; manifest split is "
                "not overwritten."
            ),
        },
        "seed": seed,
        "requested_ratios": _ratio_dict(ratios),
        "setting": setting,
        "identity_keys": {
            "content": "split_group_id",
            "trial": "record_id",
            "subject": "subject_group_id",
            "bare_subject_id_is_cross_dataset_identity": False,
        },
        "subject_namespace_method": {
            "version": SUBJECT_GROUP_VERSION,
            "canonical_namespace": "dataset_version::paradigm::subject_id",
            "id_algorithm": (
                "subject-group-v1- + first 24 hex chars of SHA-256("
                "subject-group-v1 NUL canonical_namespace)"
            ),
        },
        "deterministic_ordering": {
            "partition_assignment": (
                "content: SHA-256 threshold on seed NUL split_group_id; "
                "subjects in Setting B: cohort-local SHA-256 rank + quota"
            ),
            "all_identifier_lists": "ascending Unicode code-point order",
            "json": "UTF-8, LF, sort_keys=True, indent=2, terminal newline",
            "wall_clock_fields": "none",
        },
        "key_assumptions_and_limitations": assumptions,
    }


def _attach_selection(
    artifact: dict[str, object],
    selected: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    excluded_by_reason: Mapping[str, Sequence[Mapping[str, object]]],
    all_rows: Sequence[Mapping[str, object]],
    all_content_assignments: Mapping[str, str],
) -> None:
    partition_payload = {
        partition: _partition_payload(selected[partition]) for partition in PARTITIONS
    }
    excluded_ids_by_reason = {
        reason: sorted(str(row["record_id"]) for row in excluded_by_reason[reason])
        for reason in sorted(excluded_by_reason)
    }
    excluded_record_ids = sorted(
        record_id
        for values in excluded_ids_by_reason.values()
        for record_id in values
    )
    selected_ids = [
        record_id
        for partition in PARTITIONS
        for record_id in partition_payload[partition]["record_ids"]
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("A record_id appears in more than one selected partition")
    if len(excluded_record_ids) != len(set(excluded_record_ids)):
        raise ValueError("An excluded record_id has more than one exclusion reason")
    if set(selected_ids) & set(excluded_record_ids):
        raise ValueError("A record_id is both selected and excluded")
    if len(selected_ids) + len(excluded_record_ids) != len(all_rows):
        raise ValueError("Selected + excluded trial accounting is incomplete")

    content_counts = Counter(all_content_assignments.values())
    trial_counts = {
        partition: int(partition_payload[partition]["trial_count"])
        for partition in PARTITIONS
    }
    selected_count = sum(trial_counts.values())
    group_counts = {
        partition: int(partition_payload[partition]["content_group_count"])
        for partition in PARTITIONS
    }
    subject_counts = {
        partition: int(partition_payload[partition]["subject_group_count"])
        for partition in PARTITIONS
    }
    artifact["partitions"] = partition_payload
    artifact["excluded_record_ids"] = excluded_record_ids
    artifact["excluded_by_reason"] = excluded_ids_by_reason
    artifact["counts"] = {
        "manifest_trial_count": len(all_rows),
        "selected_trial_count": selected_count,
        "excluded_trial_count": len(excluded_record_ids),
        "trial_counts": trial_counts,
        "content_group_counts_in_selected": group_counts,
        "subject_group_counts_in_selected": subject_counts,
        "all_content_assignment_group_counts": {
            partition: content_counts.get(partition, 0) for partition in PARTITIONS
        },
        "all_content_group_count": len(all_content_assignments),
        "all_subject_group_count": len(
            {str(row["_subject_group_id"]) for row in all_rows}
        ),
    }
    artifact["actual_ratios"] = {
        "all_content_groups": {
            partition: content_counts.get(partition, 0) / len(all_content_assignments)
            for partition in PARTITIONS
        },
        "selected_trials_over_manifest": {
            partition: trial_counts[partition] / len(all_rows) for partition in PARTITIONS
        },
        "selected_trial_distribution": {
            partition: (
                trial_counts[partition] / selected_count if selected_count else 0.0
            )
            for partition in PARTITIONS
        },
    }
    artifact["content_assignment"] = _content_assignment_payload(
        all_content_assignments
    )
    content_sets = {
        partition: set(partition_payload[partition]["split_group_ids"])
        for partition in PARTITIONS
    }
    subject_sets = {
        partition: set(partition_payload[partition]["subject_group_ids"])
        for partition in PARTITIONS
    }
    artifact["leakage_checks"] = {
        "train_test_split_group_intersection_count": len(
            content_sets["train"] & content_sets["test"]
        ),
        "all_partition_split_group_intersection_count": sum(
            len(content_sets[left] & content_sets[right])
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        ),
        "train_test_subject_group_intersection_count": len(
            subject_sets["train"] & subject_sets["test"]
        ),
        "duplicate_selected_record_id_count": len(selected_ids)
        - len(set(selected_ids)),
        "duplicate_excluded_record_id_count": len(excluded_record_ids)
        - len(set(excluded_record_ids)),
        "selected_excluded_record_id_intersection_count": len(
            set(selected_ids) & set(excluded_record_ids)
        ),
        "selected_plus_excluded_equals_manifest": (
            len(selected_ids) + len(excluded_record_ids) == len(all_rows)
        ),
    }
    artifact["quality_accounting"] = {
        "policy": "quality-independent protocol construction; no quality_flag filtering",
        "selected": {
            partition: _quality_summary(selected[partition]) for partition in PARTITIONS
        },
        "excluded": _quality_summary(
            [
                row
                for reason in sorted(excluded_by_reason)
                for row in excluded_by_reason[reason]
            ]
        ),
    }


def _partition_payload(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    record_ids = sorted(str(row["record_id"]) for row in rows)
    group_ids = sorted({str(row["split_group_id"]) for row in rows})
    subject_ids = sorted({str(row["_subject_group_id"]) for row in rows})
    paradigms = Counter(str(row["paradigm"]) for row in rows)
    return {
        "record_ids": record_ids,
        "split_group_ids": group_ids,
        "subject_group_ids": subject_ids,
        "trial_count": len(record_ids),
        "content_group_count": len(group_ids),
        "subject_group_count": len(subject_ids),
        "paradigm_trial_counts": dict(sorted(paradigms.items())),
    }


def _content_assignment_payload(
    assignments: Mapping[str, str],
) -> dict[str, object]:
    split_group_ids = {
        partition: sorted(
            group_id
            for group_id, assignment in assignments.items()
            if assignment == partition
        )
        for partition in PARTITIONS
    }
    return {
        "identity_key": "split_group_id",
        "algorithm": "SHA-256 threshold on seed NUL split_group_id",
        "split_group_ids": split_group_ids,
        "content_group_counts": {
            partition: len(split_group_ids[partition]) for partition in PARTITIONS
        },
    }


def _strict_cross_paradigm_group_masks(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, set[str]]:
    masks: dict[str, set[str]] = {
        "fuzzy_global_alignment_group_excluded_from_strict_main": set(),
        "missing_text_group_excluded_from_strict_main": set(),
        "material_variant_uncertain_group_excluded_from_strict_main": set(),
    }
    for row in rows:
        group_id = str(row["split_group_id"])
        if row["global_text_alignment_status"] == "fuzzy":
            masks["fuzzy_global_alignment_group_excluded_from_strict_main"].add(
                group_id
            )
        if row["raw_text"] is None or not str(row["raw_text"]).strip():
            masks["missing_text_group_excluded_from_strict_main"].add(group_id)
        if "material_variant_uncertain" in _quality_flags(row):
            masks[
                "material_variant_uncertain_group_excluded_from_strict_main"
            ].add(group_id)
    return masks


def _mask_reason(
    group_id: str,
    masks: Mapping[str, set[str]],
) -> str | None:
    # Priority is stable and gives each excluded record exactly one reason.
    for reason in (
        "fuzzy_global_alignment_group_excluded_from_strict_main",
        "missing_text_group_excluded_from_strict_main",
        "material_variant_uncertain_group_excluded_from_strict_main",
    ):
        if group_id in masks[reason]:
            return reason
    return None


def _quality_flags(row: Mapping[str, object]) -> set[str]:
    value = row.get("quality_flag")
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item}
    return {item for item in str(value).split("|") if item}


def _quality_summary(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    materialized = list(rows)
    flags = Counter(
        flag for row in materialized for flag in sorted(_quality_flags(row))
    )
    return {
        "trial_count": len(materialized),
        "quality_flag_trial_counts": dict(sorted(flags.items())),
        "fuzzy_global_alignment_trial_count": sum(
            row["global_text_alignment_status"] == "fuzzy" for row in materialized
        ),
        "unresolved_global_alignment_trial_count": sum(
            row["global_text_alignment_status"] == "unresolved"
            for row in materialized
        ),
        "missing_text_trial_count": sum(
            row["raw_text"] is None or not str(row["raw_text"]).strip()
            for row in materialized
        ),
        "implausible_eeg_trial_duration_trial_count": sum(
            "implausible_eeg_trial_duration" in _quality_flags(row)
            for row in materialized
        ),
        "material_variant_uncertain_trial_count": sum(
            "material_variant_uncertain" in _quality_flags(row)
            for row in materialized
        ),
    }


def _ratio_dict(ratios: SplitRatios) -> dict[str, float]:
    ratios.validate()
    return {
        "train": ratios.train,
        "validation": ratios.valid,
        "test": ratios.test,
    }
