from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data.splitting import assign_content_groups


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "all_trials.parquet"
ARTIFACTS = {
    "A": ROOT / "splits" / "text_unseen_seed42.json",
    "B": ROOT / "splits" / "subject_text_unseen_seed42.json",
    "C": ROOT / "splits" / "cross_paradigm_seed42.json",
}

pytestmark = pytest.mark.integration


def _load(path: Path) -> dict[str, object]:
    if not MANIFEST.exists():
        pytest.skip("Local real manifest is unavailable")
    assert path.exists(), f"Generate protocol artifacts first: missing {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_record_accounting(
    payload: dict[str, object],
    *,
    manifest_count: int,
) -> None:
    selected = [
        record_id
        for partition in ("train", "validation", "test")
        for record_id in payload["partitions"][partition]["record_ids"]
    ]
    excluded = payload["excluded_record_ids"]
    assert len(selected) == len(set(selected))
    assert len(excluded) == len(set(excluded))
    assert not set(selected) & set(excluded)
    assert len(selected) + len(excluded) == manifest_count


def test_real_setting_a_content_is_unseen_and_accounting_is_complete() -> None:
    artifact = _load(ARTIFACTS["A"])
    assert artifact["manifest_row_count"] == 168_156
    train = set(artifact["partitions"]["train"]["split_group_ids"])
    test = set(artifact["partitions"]["test"]["split_group_ids"])
    assert not train & test
    assert artifact["seed"] == 42
    _assert_record_accounting(artifact, manifest_count=168_156)


def test_real_different_seed_changes_content_group_assignments() -> None:
    artifact = _load(ARTIFACTS["A"])
    stored = {
        group_id: partition
        for partition, group_ids in artifact["content_assignment"][
            "split_group_ids"
        ].items()
        for group_id in group_ids
    }
    seed_43 = assign_content_groups(stored, seed=43)
    changed = sum(
        stored[group_id]
        != ("validation" if partition == "valid" else partition)
        for group_id, partition in seed_43.items()
    )
    assert changed > 0


def test_real_setting_b_has_strict_subject_and_content_holdout() -> None:
    artifact = _load(ARTIFACTS["B"])
    train_content = set(artifact["partitions"]["train"]["split_group_ids"])
    test_content = set(artifact["partitions"]["test"]["split_group_ids"])
    train_subject = set(artifact["partitions"]["train"]["subject_group_ids"])
    test_subject = set(artifact["partitions"]["test"]["subject_group_ids"])
    assert not train_content & test_content
    assert not train_subject & test_subject
    assert artifact["counts"]["excluded_trial_count"] > 0
    _assert_record_accounting(artifact, manifest_count=168_156)


def test_real_cross_paradigm_protocols_are_zero_shot_and_content_disjoint() -> None:
    artifact = _load(ARTIFACTS["C"])
    assert set(artifact["protocols"]) == {
        "pl_to_silent_reading_unseen_text",
        "silent_reading_to_pl_unseen_text",
        "pl_silent_reading_to_ra_unseen_text",
    }
    for protocol in artifact["protocols"].values():
        target = protocol["target_paradigm"]
        assert (
            protocol["partitions"]["train"]["paradigm_trial_counts"].get(target, 0)
            == 0
        )
        assert (
            protocol["partitions"]["validation"]["paradigm_trial_counts"].get(
                target, 0
            )
            == 0
        )
        train_content = set(protocol["partitions"]["train"]["split_group_ids"])
        test_content = set(protocol["partitions"]["test"]["split_group_ids"])
        assert not train_content & test_content
        _assert_record_accounting(protocol, manifest_count=168_156)


@pytest.mark.parametrize("path", ARTIFACTS.values(), ids=ARTIFACTS.keys())
def test_real_artifact_is_canonical_and_matches_reported_sha(path: Path) -> None:
    artifact = _load(path)
    canonical = (
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert path.read_bytes() == canonical
    report = json.loads(
        (ROOT / "reports" / "split_protocol_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert hashlib.sha256(canonical).hexdigest() == report["artifact_sha256"][
        path.name
    ]
