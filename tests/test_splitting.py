import pytest

from data.splitting import (
    SplitRatios,
    assert_group_split_integrity,
    assign_content_groups,
    assign_split,
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

