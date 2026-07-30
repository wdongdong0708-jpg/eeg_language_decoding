"""Candidate-pool construction with explicit length matching."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateMetadata:
    candidate_id: str
    character_count: int
    duration_sec: float | None = None


@dataclass(frozen=True, slots=True)
class LengthMatchPolicy:
    character_count_tolerance: int = 1
    duration_tolerance_sec: float | None = 0.5
    minimum_pool_size: int = 20

    def validate(self) -> None:
        if self.character_count_tolerance < 0:
            raise ValueError("character_count_tolerance cannot be negative")
        if self.duration_tolerance_sec is not None and self.duration_tolerance_sec < 0:
            raise ValueError("duration_tolerance_sec cannot be negative")
        if self.minimum_pool_size < 2:
            raise ValueError("minimum_pool_size must be at least 2")


def length_matched_candidates(
    query: CandidateMetadata,
    candidates: list[CandidateMetadata],
    *,
    policy: LengthMatchPolicy,
) -> list[CandidateMetadata]:
    """Return a strict pool; never silently relax matching tolerances."""

    policy.validate()
    pool: list[CandidateMetadata] = []
    for candidate in candidates:
        if abs(candidate.character_count - query.character_count) > (
            policy.character_count_tolerance
        ):
            continue
        if (
            policy.duration_tolerance_sec is not None
            and query.duration_sec is not None
            and candidate.duration_sec is not None
            and abs(candidate.duration_sec - query.duration_sec)
            > policy.duration_tolerance_sec
        ):
            continue
        pool.append(candidate)

    if not any(candidate.candidate_id == query.candidate_id for candidate in pool):
        raise ValueError("Length-matched pool does not contain the positive candidate")
    if len(pool) < policy.minimum_pool_size:
        raise ValueError(
            f"Length-matched pool has {len(pool)} candidates; "
            f"minimum is {policy.minimum_pool_size}"
        )
    return pool

