"""Rank-based retrieval metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import median


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    median_rank: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalMetricsAggregate:
    """Mean and standard deviation for a collection of metric records."""

    count: int
    mean: RetrievalMetrics
    std: RetrievalMetrics


def metrics_from_ranks(ranks: Sequence[int]) -> RetrievalMetrics:
    if not ranks:
        raise ValueError("ranks cannot be empty")
    if any(rank < 1 for rank in ranks):
        raise ValueError("Ranks are one-based and must be positive")
    n = len(ranks)
    return RetrievalMetrics(
        recall_at_1=sum(rank <= 1 for rank in ranks) / n,
        recall_at_5=sum(rank <= 5 for rank in ranks) / n,
        recall_at_10=sum(rank <= 10 for rank in ranks) / n,
        median_rank=float(median(ranks)),
        mean_reciprocal_rank=sum(1.0 / rank for rank in ranks) / n,
    )


def metrics_by_group(
    ranks: Sequence[int],
    group_ids: Sequence[str],
) -> dict[str, RetrievalMetrics]:
    """Compute retrieval metrics independently for each query group."""

    if len(ranks) != len(group_ids):
        raise ValueError("ranks and group_ids must have the same length")
    if not ranks:
        raise ValueError("ranks cannot be empty")
    grouped: dict[str, list[int]] = {}
    for rank, group_id in zip(ranks, group_ids, strict=True):
        grouped.setdefault(str(group_id), []).append(rank)
    return {
        group_id: metrics_from_ranks(grouped[group_id])
        for group_id in sorted(grouped)
    }


def aggregate_retrieval_metrics(
    records: Sequence[RetrievalMetrics],
    *,
    ddof: int = 1,
) -> RetrievalMetricsAggregate:
    """Aggregate metric records using a configurable standard-deviation ddof.

    BrainMagick's paper notebook uses pandas ``std()``, whose default is the
    sample standard deviation (``ddof=1``). A singleton collection has a
    reported standard deviation of zero here so JSON output remains finite.
    """

    if not records:
        raise ValueError("records cannot be empty")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    names = tuple(RetrievalMetrics.__dataclass_fields__)
    means: dict[str, float] = {}
    standard_deviations: dict[str, float] = {}
    for name in names:
        values = [float(getattr(record, name)) for record in records]
        value_mean = sum(values) / len(values)
        means[name] = value_mean
        if len(values) <= ddof:
            standard_deviations[name] = 0.0
        else:
            variance = sum((value - value_mean) ** 2 for value in values) / (
                len(values) - ddof
            )
            standard_deviations[name] = sqrt(variance)
    return RetrievalMetricsAggregate(
        count=len(records),
        mean=RetrievalMetrics(**means),
        std=RetrievalMetrics(**standard_deviations),
    )



def expected_random_retrieval_metrics(
    candidate_count: int,
) -> RetrievalMetrics:
    """Analytical chance metrics for a uniformly random candidate ordering."""

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    return RetrievalMetrics(
        recall_at_1=1.0 / candidate_count,
        recall_at_5=min(5, candidate_count) / candidate_count,
        recall_at_10=min(10, candidate_count) / candidate_count,
        median_rank=(candidate_count + 1) / 2.0,
        mean_reciprocal_rank=(
            sum(1.0 / rank for rank in range(1, candidate_count + 1))
            / candidate_count
        ),
    )
