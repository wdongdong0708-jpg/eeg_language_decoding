"""Metadata shortcut controls for unique-text seen-text retrieval."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evaluation.retrieval_metrics import (
    RetrievalMetrics,
    expected_random_retrieval_metrics,
    metrics_from_ranks,
)


@dataclass(frozen=True, slots=True)
class SeenTextShortcutResult:
    metrics: Mapping[str, RetrievalMetrics]
    query_count: int
    candidate_count: int


def _tie_hash(seed: int, query_id: str, candidate_id: str, baseline: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{baseline}\0{query_id}\0{candidate_id}".encode("utf-8")
    ).digest()


def _fixed_order_ranks(
    queries: Sequence[Mapping[str, object]],
    ordered_candidate_ids: Sequence[str],
) -> list[int]:
    rank_by_candidate = {
        candidate_id: rank
        for rank, candidate_id in enumerate(ordered_candidate_ids, start=1)
    }
    return [rank_by_candidate[str(row["span_text_id"])] for row in queries]


def evaluate_seen_text_shortcuts(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    *,
    semantic_only: bool,
    seed: int,
) -> SeenTextShortcutResult:
    queries = [
        row for row in test_rows if not semantic_only or bool(row["is_semantic_unit"])
    ]
    candidate_ids = sorted(
        {
            str(row["span_text_id"])
            for row in test_rows
            if not semantic_only or bool(row["is_semantic_unit"])
        }
    )
    candidate_set = set(candidate_ids)
    if not queries or not candidate_ids:
        raise ValueError("Shortcut evaluation requires queries and candidates")
    train_frequency = Counter(
        str(row["span_text_id"])
        for row in train_rows
        if str(row["span_text_id"]) in candidate_set
    )
    frequency_order = sorted(
        candidate_ids,
        key=lambda candidate_id: (-train_frequency[candidate_id], candidate_id),
    )
    frequency_ranks = _fixed_order_ranks(queries, frequency_order)

    train_subject_frequency: Counter[tuple[str, str]] = Counter(
        (str(row["subject_group_id"]), str(row["span_text_id"]))
        for row in train_rows
        if str(row["span_text_id"]) in candidate_set
    )
    subject_orders: dict[str, list[str]] = {}
    for subject_id in sorted({str(row["subject_group_id"]) for row in queries}):
        subject_orders[subject_id] = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -train_subject_frequency[(subject_id, candidate_id)],
                candidate_id,
            ),
        )
    subject_rank_maps = {
        subject_id: {
            candidate_id: rank
            for rank, candidate_id in enumerate(order, start=1)
        }
        for subject_id, order in subject_orders.items()
    }
    subject_ranks = [
        subject_rank_maps[str(row["subject_group_id"])][str(row["span_text_id"])]
        for row in queries
    ]

    candidates_by_position: dict[int, set[str]] = defaultdict(set)
    for row in test_rows:
        candidate_id = str(row["span_text_id"])
        if candidate_id in candidate_set:
            candidates_by_position[int(row["stimulus_position"])].add(candidate_id)
    position_ranks: list[int] = []
    for row in queries:
        query_id = (
            f"{row['record_id']}::{row['span_event_id']}::{row['span_start_clock']}"
        )
        candidate_id = str(row["span_text_id"])
        tied = sorted(
            candidates_by_position[int(row["stimulus_position"])],
            key=lambda value: _tie_hash(seed, query_id, value, "position"),
        )
        position_ranks.append(tied.index(candidate_id) + 1)

    random_metrics = expected_random_retrieval_metrics(len(candidate_ids))
    return SeenTextShortcutResult(
        metrics={
            "random_analytical": random_metrics,
            "duration_only_fixed_length_tie": random_metrics,
            "character_count_only_fixed_k_tie": random_metrics,
            "padding_mask_only_no_mask_exposed": random_metrics,
            "train_frequency": metrics_from_ranks(frequency_ranks),
            "subject_id_train_frequency": metrics_from_ranks(subject_ranks),
            "sentence_position_test_occurrence_oracle": metrics_from_ranks(
                position_ranks
            ),
        },
        query_count=len(queries),
        candidate_count=len(candidate_ids),
    )
