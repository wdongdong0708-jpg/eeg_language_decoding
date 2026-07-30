"""Deterministic content-group splitting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Literal

SplitName = Literal["train", "valid", "test"]


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.8
    valid: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        values = (self.train, self.valid, self.test)
        if any(value < 0 for value in values):
            raise ValueError("Split ratios cannot be negative")
        if abs(sum(values) - 1.0) > 1e-12:
            raise ValueError(f"Split ratios must sum to 1, got {sum(values)}")
        if self.train == 0 or self.test == 0:
            raise ValueError("Train and test fractions must be non-zero")


def stable_hash_fraction(content_id: str, *, seed: int | str) -> float:
    """Map a content ID to [0, 1) using a platform-independent SHA-256 hash."""

    if not content_id:
        raise ValueError("content_id cannot be empty")
    payload = f"{seed}\0{content_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def assign_split(
    content_id: str,
    *,
    seed: int | str,
    ratios: SplitRatios = SplitRatios(),
) -> SplitName:
    ratios.validate()
    fraction = stable_hash_fraction(content_id, seed=seed)
    if fraction < ratios.train:
        return "train"
    if fraction < ratios.train + ratios.valid:
        return "valid"
    return "test"


def assign_content_groups(
    content_ids: Iterable[str],
    *,
    seed: int | str,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, SplitName]:
    """Assign each unique content group exactly once."""

    return {
        content_id: assign_split(content_id, seed=seed, ratios=ratios)
        for content_id in sorted(set(content_ids))
    }


def quota_counts(
    group_count: int,
    *,
    ratios: SplitRatios = SplitRatios(),
    require_validation_and_test: bool = True,
) -> dict[SplitName, int]:
    """Choose deterministic integer quotas closest to requested group ratios.

    This is intended for small subject cohorts, where independent hash thresholds
    can leave validation or test empty. Quotas are selected at the group level;
    trial counts are never used.
    """

    ratios.validate()
    if group_count < 0:
        raise ValueError("group_count cannot be negative")
    if group_count == 0:
        return {"train": 0, "valid": 0, "test": 0}
    if require_validation_and_test and group_count < 3:
        raise ValueError(
            "At least three groups are required to keep train, validation and test non-empty"
        )

    requested = {
        "train": ratios.train * group_count,
        "valid": ratios.valid * group_count,
        "test": ratios.test * group_count,
    }
    minimum = {
        "train": 1,
        "valid": int(require_validation_and_test),
        "test": int(require_validation_and_test),
    }
    candidates: list[tuple[float, float, int, int, int]] = []
    for train_count in range(minimum["train"], group_count + 1):
        for valid_count in range(minimum["valid"], group_count - train_count + 1):
            test_count = group_count - train_count - valid_count
            if test_count < minimum["test"]:
                continue
            absolute_error = sum(
                abs(observed - requested[name])
                for name, observed in (
                    ("train", train_count),
                    ("valid", valid_count),
                    ("test", test_count),
                )
            )
            squared_error = sum(
                math.pow(observed - requested[name], 2)
                for name, observed in (
                    ("train", train_count),
                    ("valid", valid_count),
                    ("test", test_count),
                )
            )
            # Prefer train, then validation, on exact objective ties.
            candidates.append(
                (
                    absolute_error,
                    squared_error,
                    -train_count,
                    -valid_count,
                    test_count,
                )
            )
    if not candidates:
        raise ValueError("No feasible quota allocation")
    _, _, negative_train, negative_valid, test_count = min(candidates)
    return {
        "train": -negative_train,
        "valid": -negative_valid,
        "test": test_count,
    }


def assign_groups_by_quota(
    group_ids: Iterable[str],
    *,
    seed: int | str,
    ratios: SplitRatios = SplitRatios(),
    require_validation_and_test: bool = True,
) -> dict[str, SplitName]:
    """Hash-rank unique groups, then allocate deterministic integer quotas."""

    unique_ids = sorted(set(group_ids))
    if any(not group_id for group_id in unique_ids):
        raise ValueError("group IDs cannot be empty")
    quotas = quota_counts(
        len(unique_ids),
        ratios=ratios,
        require_validation_and_test=require_validation_and_test,
    )
    ordered = sorted(
        unique_ids,
        key=lambda group_id: (stable_hash_fraction(group_id, seed=seed), group_id),
    )
    train_end = quotas["train"]
    valid_end = train_end + quotas["valid"]
    return {
        group_id: (
            "train"
            if index < train_end
            else "valid"
            if index < valid_end
            else "test"
        )
        for index, group_id in enumerate(ordered)
    }


def assert_group_split_integrity(
    rows: Iterable[dict[str, object]],
    *,
    group_key: str = "content_id",
    split_key: str = "split",
) -> None:
    """Reject manifests where one content group appears in multiple splits."""

    by_group: dict[str, set[str]] = {}
    for row in rows:
        group = str(row[group_key])
        split = str(row[split_key])
        by_group.setdefault(group, set()).add(split)
    violations = {group: values for group, values in by_group.items() if len(values) != 1}
    if violations:
        preview = dict(list(violations.items())[:5])
        raise ValueError(f"Group split leakage detected: {preview}")
