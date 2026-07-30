"""Deterministic content-group splitting."""

from __future__ import annotations

import hashlib
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

