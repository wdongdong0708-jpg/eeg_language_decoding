from __future__ import annotations

from pathlib import Path

from data.protocol_splitting import (
    build_cross_paradigm_protocol,
    build_subject_text_unseen_protocol,
    build_text_unseen_protocol,
    make_subject_group_id,
    write_json_deterministic,
)


def _synthetic_rows() -> list[dict[str, object]]:
    cohorts = (
        ("ChineseEEG1", "silent_reading", ("01", "02", "03")),
        ("ChineseEEG2", "passive_listening", ("01", "02", "03")),
        ("ChineseEEG2", "reading_aloud", ("r1", "r2", "r3")),
    )
    rows: list[dict[str, object]] = []
    index = 0
    for dataset, paradigm, subjects in cohorts:
        for subject in subjects:
            for content_index in range(40):
                index += 1
                status = "fuzzy" if content_index == 0 else "exact"
                raw_text = None if content_index == 1 else f"文本{content_index}"
                quality_flag = (
                    "material_variant_uncertain"
                    if content_index == 2
                    else "implausible_eeg_trial_duration"
                    if content_index == 3
                    else "ok"
                )
                rows.append(
                    {
                        "record_id": f"record-{index:05d}",
                        "sentence_id": f"sentence-{content_index:03d}",
                        "global_text_id": f"global-{content_index:03d}",
                        "normalized_text_hash": f"hash-{content_index:03d}",
                        "split_group_id": f"content-{content_index:03d}",
                        "split": "train",
                        "eeg_start_sample": content_index * 100,
                        "eeg_end_sample": content_index * 100 + 90,
                        "dataset_version": dataset,
                        "paradigm": paradigm,
                        "subject_id": subject,
                        "raw_text": raw_text,
                        "global_text_alignment_status": status,
                        "quality_flag": quality_flag,
                        "manifest_schema_version": "trial-manifest-v1",
                        "split_seed": 20260730,
                    }
                )
    return rows


def _partition_sets(
    artifact: dict[str, object],
    key: str,
) -> dict[str, set[str]]:
    return {
        partition: set(artifact["partitions"][partition][key])
        for partition in ("train", "validation", "test")
    }


def _assert_complete_accounting(artifact: dict[str, object]) -> None:
    selected = [
        record_id
        for partition in ("train", "validation", "test")
        for record_id in artifact["partitions"][partition]["record_ids"]
    ]
    excluded = artifact["excluded_record_ids"]
    assert len(selected) == len(set(selected))
    assert len(excluded) == len(set(excluded))
    assert not set(selected) & set(excluded)
    assert len(selected) + len(excluded) == artifact["manifest_row_count"]
    assert artifact["leakage_checks"]["selected_plus_excluded_equals_manifest"]


def test_setting_a_is_content_grouped_and_subjects_can_overlap() -> None:
    artifact = build_text_unseen_protocol(_synthetic_rows(), seed=42)
    contents = _partition_sets(artifact, "split_group_ids")
    assert not contents["train"] & contents["test"]
    assert not contents["train"] & contents["validation"]
    assert not contents["validation"] & contents["test"]
    assert artifact["subject_overlap"]["train_test_subject_overlap_count"] > 0
    _assert_complete_accounting(artifact)


def test_setting_b_strict_diagonal_has_no_subject_or_content_leakage() -> None:
    artifact = build_subject_text_unseen_protocol(_synthetic_rows(), seed=42)
    contents = _partition_sets(artifact, "split_group_ids")
    subjects = _partition_sets(artifact, "subject_group_ids")
    assert not contents["train"] & contents["test"]
    assert not subjects["train"] & subjects["test"]
    assert artifact["excluded_record_ids"]
    assert all(
        reason.startswith("off_diagonal_subject_")
        for reason in artifact["excluded_by_reason"]
    )
    _assert_complete_accounting(artifact)


def test_setting_c_is_zero_shot_and_unseen_text() -> None:
    artifact = build_cross_paradigm_protocol(_synthetic_rows(), seed=42)
    assert len(artifact["protocols"]) == 3
    for protocol in artifact["protocols"].values():
        target = protocol["target_paradigm"]
        for partition in ("train", "validation"):
            assert (
                protocol["partitions"][partition]["paradigm_trial_counts"].get(
                    target, 0
                )
                == 0
            )
        contents = _partition_sets(protocol, "split_group_ids")
        assert not contents["train"] & contents["test"]
        _assert_complete_accounting(
            {
                **protocol,
                "manifest_row_count": artifact["manifest_row_count"],
            }
        )
        assert (
            protocol["leakage_checks"][
                "target_paradigm_in_train_or_validation_count"
            ]
            == 0
        )


def test_subject_identity_is_namespaced_by_dataset_and_cohort() -> None:
    first = {
        "dataset_version": "ChineseEEG1",
        "paradigm": "silent_reading",
        "subject_id": "04",
    }
    second = {
        "dataset_version": "ChineseEEG2",
        "paradigm": "passive_listening",
        "subject_id": "04",
    }
    assert make_subject_group_id(first) != make_subject_group_id(second)


def test_protocol_json_is_byte_deterministic(tmp_path: Path) -> None:
    rows = _synthetic_rows()
    first = build_subject_text_unseen_protocol(rows, seed=42)
    second = build_subject_text_unseen_protocol(list(rows), seed=42)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_hash = write_json_deterministic(first_path, first)
    second_hash = write_json_deterministic(second_path, second)
    assert first_hash == second_hash
    assert first_path.read_bytes() == second_path.read_bytes()


def test_different_seed_changes_content_assignment() -> None:
    rows = _synthetic_rows()
    first = build_text_unseen_protocol(rows, seed=42)
    second = build_text_unseen_protocol(rows, seed=43)
    first_train = set(first["partitions"]["train"]["split_group_ids"])
    second_train = set(second["partitions"]["train"]["split_group_ids"])
    assert first_train != second_train
