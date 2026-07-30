"""Compact audit reports for versioned protocol split artifacts."""

from __future__ import annotations

from collections import Counter
from typing import Mapping


def build_protocol_audit(
    artifacts: Mapping[str, Mapping[str, object]],
    *,
    artifact_sha256: Mapping[str, str],
) -> dict[str, object]:
    setting_a = artifacts[next(name for name in artifacts if name.startswith("text_unseen_"))]
    setting_b = artifacts[
        next(name for name in artifacts if name.startswith("subject_text_unseen_"))
    ]
    setting_c = artifacts[
        next(name for name in artifacts if name.startswith("cross_paradigm_"))
    ]
    return {
        "protocol_version": setting_a["protocol_version"],
        "manifest_path": setting_a["manifest_path"],
        "manifest_row_count": setting_a["manifest_row_count"],
        "manifest_critical_field_fingerprint": setting_a[
            "manifest_critical_field_fingerprint"
        ],
        "seed": setting_a["seed"],
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "setting_a": _single_summary(setting_a),
        "setting_b": {
            **_single_summary(setting_b),
            "off_diagonal_exclusion_counts": {
                reason: len(record_ids)
                for reason, record_ids in setting_b["excluded_by_reason"].items()
            },
            "subject_cohorts": setting_b["subject_assignment"]["cohorts"],
        },
        "setting_c": {
            "strict_sensitivity_masks": setting_c["strict_sensitivity_masks"],
            "protocols": {
                name: {
                    "source_paradigms": protocol["source_paradigms"],
                    "target_paradigm": protocol["target_paradigm"],
                    **_single_summary(protocol),
                }
                for name, protocol in setting_c["protocols"].items()
            },
        },
        "scientific_policy": {
            "quality_filtering": (
                "None. Protocol construction is independent of quality_flag."
            ),
            "fuzzy_global_alignment": (
                "Retained in Settings A/B accounting and assignment; whole implicated "
                "content groups excluded from Setting C strict main protocols."
            ),
            "missing_text": (
                "Retained in Settings A/B accounting and assignment; whole implicated "
                "content groups excluded from Setting C strict main protocols."
            ),
            "material_variant_uncertain": (
                "Retained in Settings A/B accounting and assignment; whole implicated "
                "content groups excluded from Setting C strict main protocols."
            ),
            "implausible_eeg_trial_duration": (
                "Retained everywhere unless excluded by a protocol cell/mask unrelated "
                "to duration; never used to assign a partition."
            ),
            "unresolved_global_alignment": (
                "Retained. Zero-shot transfer does not require paired source/target "
                "trials; unresolved cross-dataset variants remain a scientific risk."
            ),
        },
        "reproducibility_contract": {
            "identifier_lists_sorted": True,
            "json_keys_sorted": True,
            "wall_clock_fields_absent": True,
            "manifest_embedded_split_overwritten": False,
            "expected_rebuild_behavior": (
                "Same manifest bytes, seed, schema and implementation produce identical "
                "artifact SHA-256."
            ),
        },
    }


def render_protocol_audit_markdown(audit: Mapping[str, object]) -> str:
    a = audit["setting_a"]
    b = audit["setting_b"]
    c = audit["setting_c"]
    lines = [
        "# Split Protocol Audit",
        "",
        "This audit covers protocol-level assignments only. It does not generate "
        "windows, embeddings, models, or retrieval results.",
        "",
        "## Immutable input contract",
        "",
        f"- Manifest: `{audit['manifest_path']}`",
        f"- Rows: {audit['manifest_row_count']:,}",
        f"- Critical-field SHA-256: "
        f"`{audit['manifest_critical_field_fingerprint']['sha256']}`",
        f"- Protocol seed: `{audit['seed']}`",
        "- The manifest's embedded split is preserved and is not overwritten.",
        "",
        "## Artifact fingerprints",
        "",
        "| artifact | SHA-256 |",
        "|---|---|",
    ]
    for name, digest in audit["artifact_sha256"].items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## Setting A — unseen text, subjects visible",
            "",
            _count_table_header(),
            _count_table_row("A", a),
            "",
            f"- Train/test content overlap: "
            f"{a['leakage_checks']['train_test_split_group_intersection_count']}",
            f"- Train/test subject overlap (allowed): "
            f"{a['leakage_checks']['train_test_subject_group_intersection_count']}",
            f"- Content-group ratios (train/validation/test): "
            f"{_format_ratios(a['actual_ratios']['all_content_groups'])}",
            f"- Trial ratios (train/validation/test): "
            f"{_format_ratios(a['actual_ratios']['selected_trial_distribution'])}",
            "",
            "## Setting B — unseen subjects and unseen text",
            "",
            _count_table_header(),
            _count_table_row("B strict diagonal", b),
            "",
            f"- Train/test content overlap: "
            f"{b['leakage_checks']['train_test_split_group_intersection_count']}",
            f"- Train/test subject overlap: "
            f"{b['leakage_checks']['train_test_subject_group_intersection_count']}",
            f"- Off-diagonal excluded trials: {b['counts']['excluded_trial_count']:,}",
            f"- Content-group assignment ratios (train/validation/test): "
            f"{_format_ratios(b['actual_ratios']['all_content_groups'])}",
            f"- Selected-trial distribution (train/validation/test): "
            f"{_format_ratios(b['actual_ratios']['selected_trial_distribution'])}",
            "",
            "### Subject cohort quotas",
            "",
            "| cohort | subjects | train | validation | test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cohort in b["subject_cohorts"]:
        counts = cohort["assigned_counts"]
        lines.append(
            f"| {cohort['cohort']} | {cohort['subject_group_count']} | "
            f"{counts['train']} | {counts['validation']} | {counts['test']} |"
        )
    lines.extend(
        [
            "",
            "### Off-diagonal exclusions",
            "",
            "| reason | trials |",
            "|---|---:|",
        ]
    )
    for reason, count in b["off_diagonal_exclusion_counts"].items():
        lines.append(f"| `{reason}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Setting C — zero-shot cross-paradigm, unseen text",
            "",
            _count_table_header(),
        ]
    )
    for name, protocol in c["protocols"].items():
        label = (
            f"{name}: {','.join(protocol['source_paradigms'])}"
            f" → {protocol['target_paradigm']}"
        )
        lines.append(_count_table_row(label, protocol))
    lines.extend(["", "Selected-trial train/validation/test distributions:"])
    for name, protocol in c["protocols"].items():
        lines.append(
            f"- `{name}`: "
            f"{_format_ratios(protocol['actual_ratios']['selected_trial_distribution'])}"
        )
    lines.extend(
        [
            "",
            "Every main protocol has zero target-paradigm trials in train/validation "
            "and zero train/test content overlap.",
            "",
            "### Strict identity sensitivity masks",
            "",
            "| mask | content groups | manifest trials in groups |",
            "|---|---:|---:|",
        ]
    )
    for reason, details in c["strict_sensitivity_masks"]["masks"].items():
        lines.append(
            f"| `{reason}` | {details['content_group_count']:,} | "
            f"{details['trial_count_in_manifest']:,} |"
        )
    lines.extend(
        [
            "",
            "## Quality and alignment treatment",
            "",
        ]
    )
    for name, policy in audit["scientific_policy"].items():
        lines.append(f"- `{name}`: {policy}")
    lines.extend(
        [
            "",
            "## Remaining scientific risks",
            "",
            "- A deterministic content split prevents exact `split_group_id` leakage, "
            "but near-duplicate or semantically equivalent text not merged by the "
            "manifest can still cross partitions.",
            "- Trial counts need not follow 80/10/10 because assignment is group-level "
            "and repeated content has unequal numbers of trials.",
            "- Setting B's strict diagonal excludes most off-diagonal observations; "
            "reported performance therefore conditions on both held-out subject and "
            "held-out content quotas.",
            "- Cross-paradigm cohorts are disjoint and acquisition/paradigm shifts are "
            "confounded with dataset and subject cohort.",
            "- Unresolved global alignment is retained; strict conclusions should be "
            "paired with a sensitivity analysis that masks it.",
            "- Protocol construction alone does not remove duration, position, padding, "
            "subject, or audio-envelope shortcuts; those controls belong to later "
            "evaluation stages.",
            "",
        ]
    )
    return "\n".join(lines)


def _single_summary(artifact: Mapping[str, object]) -> dict[str, object]:
    return {
        "counts": artifact["counts"],
        "actual_ratios": artifact["actual_ratios"],
        "leakage_checks": artifact["leakage_checks"],
        "quality_accounting": artifact["quality_accounting"],
        "excluded_reason_counts": dict(
            sorted(
                Counter(
                    {
                        reason: len(record_ids)
                        for reason, record_ids in artifact[
                            "excluded_by_reason"
                        ].items()
                    }
                ).items()
            )
        ),
    }


def _count_table_header() -> str:
    return (
        "| protocol | train trials/groups/subjects | validation trials/groups/subjects "
        "| test trials/groups/subjects | excluded |"
        "\n|---|---:|---:|---:|---:|"
    )


def _count_table_row(label: str, summary: Mapping[str, object]) -> str:
    counts = summary["counts"]
    trials = counts["trial_counts"]
    groups = counts["content_group_counts_in_selected"]
    subjects = counts["subject_group_counts_in_selected"]
    cells = [
        f"{trials[partition]:,}/{groups[partition]:,}/{subjects[partition]:,}"
        for partition in ("train", "validation", "test")
    ]
    return (
        f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | "
        f"{counts['excluded_trial_count']:,} |"
    )


def _format_ratios(ratios: Mapping[str, float]) -> str:
    return "/".join(
        f"{100 * ratios[partition]:.3f}%"
        for partition in ("train", "validation", "test")
    )
