"""Rank-based retrieval metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    median_rank: float
    mean_reciprocal_rank: float


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

