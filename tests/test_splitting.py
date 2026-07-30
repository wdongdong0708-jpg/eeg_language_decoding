import pytest

from data.splitting import (
    SplitRatios,
    assign_groups_by_quota,
    assert_group_split_integrity,
    assign_content_groups,
    assign_split,
    quota_counts,
)


def test_split_is_deterministic_and_content_only() -> None:
    content_id = "content-v1-0123456789abcdef"
    assert assign_split(content_id, seed=7) == assign_split(content_id, seed=7)


def test_duplicate_content_ids_receive_one_assignment() -> None:
    assignments = assign_content_groups(["a", "b", "a"], seed=11)
    assert set(assignments) == {"a", "b"}


def test_ratio_validation() -> None:
    with pytest.raises(ValueError):
        SplitRatios(train=0.8, valid=0.2, test=0.2).validate()


def test_group_leakage_is_rejected() -> None:
    rows = [
        {"content_id": "same", "split": "train"},
        {"content_id": "same", "split": "test"},
    ]
    with pytest.raises(ValueError, match="leakage"):
        assert_group_split_integrity(rows)


@pytest.mark.parametrize(
    ("group_count", "expected"),
    [
        (4, {"train": 2, "valid": 1, "test": 1}),
        (8, {"train": 6, "valid": 1, "test": 1}),
        (10, {"train": 8, "valid": 1, "test": 1}),
    ],
)
def test_small_cohort_quota_keeps_validation_and_test_nonempty(
    group_count: int,
    expected: dict[str, int],
) -> None:
    assert quota_counts(group_count) == expected


def test_quota_assignment_is_deterministic_and_seed_sensitive() -> None:
    group_ids = [f"subject-{index}" for index in range(10)]
    first = assign_groups_by_quota(group_ids, seed=42)
    second = assign_groups_by_quota(reversed(group_ids), seed=42)
    changed = assign_groups_by_quota(group_ids, seed=43)
    assert first == second
    assert first != changed
