"""Book-stratified and book-balanced ChineseEEG1 retrieval evaluation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.chineseeeg1_span_dataset import collate_fixed_character_spans
from evaluation.retrieval_metrics import (
    RetrievalMetrics,
    expected_random_retrieval_metrics,
    metrics_from_ranks,
)

BOOK_IDS = ("garnettdream", "littleprince")


@dataclass(slots=True)
class RetrievalProtocol:
    name: str
    family: str
    candidate_ids: tuple[str, ...]
    query_indices: tuple[int, ...]
    query_universe_indices: tuple[int, ...]
    candidate_selection: Mapping[str, object]
    query_selection: Mapping[str, object]


def _stable_digest(seed: int, label: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{value}".encode("utf-8")).digest()


def _query_id(row: Mapping[str, object]) -> str:
    return f"{row['record_id']}::{row['span_event_id']}::{row['span_start_clock']}"


def _frequency_order(
    rows: Sequence[Mapping[str, object]],
    *,
    book_id: str | None = None,
    semantic_only: bool = True,
) -> tuple[str, ...]:
    counts = Counter(
        str(row["span_text_id"])
        for row in rows
        if (book_id is None or str(row["book_id"]) == book_id)
        and (not semantic_only or bool(row["is_semantic_unit"]))
    )
    return tuple(sorted(counts, key=lambda text_id: (-counts[text_id], text_id)))


def balanced_unique_candidates(
    candidates_by_book: Mapping[str, Sequence[str]],
    *,
    quota_per_book: int,
    seed: int,
    label: str,
    preserve_input_order: bool,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Select equal unique candidate contributions, deduplicating shared text."""

    if quota_per_book < 1:
        raise ValueError("quota_per_book must be positive")
    selected: set[str] = set()
    contributed: dict[str, tuple[str, ...]] = {}
    # Minority book first keeps every minority candidate eligible in full-pool
    # balancing. Ties use the stable book ID.
    order = sorted(BOOK_IDS, key=lambda book: (len(candidates_by_book[book]), book))
    output: list[str] = []
    for book in order:
        candidates = list(dict.fromkeys(str(value) for value in candidates_by_book[book]))
        if not preserve_input_order:
            candidates.sort(key=lambda value: (_stable_digest(seed, label, value), value))
        local = []
        for candidate in candidates:
            if candidate in selected:
                continue
            selected.add(candidate)
            local.append(candidate)
            output.append(candidate)
            if len(local) == quota_per_book:
                break
        if len(local) != quota_per_book:
            raise ValueError(
                f"Book {book} cannot provide {quota_per_book} unique candidates"
            )
        contributed[book] = tuple(local)
    return tuple(output), contributed


def balance_query_indices(
    rows: Sequence[Mapping[str, object]],
    eligible_indices: Sequence[int],
    *,
    seed: int,
    label: str,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """Deterministically downsample to equal query occurrence counts per book."""

    by_book: dict[str, list[int]] = {book: [] for book in BOOK_IDS}
    for index in eligible_indices:
        by_book[str(rows[index]["book_id"])].append(int(index))
    target = min(len(by_book[book]) for book in BOOK_IDS)
    if target < 1:
        raise ValueError(f"Balanced query protocol {label} has an empty book")
    selected: list[int] = []
    for book in BOOK_IDS:
        ordered = sorted(
            by_book[book],
            key=lambda index: (
                _stable_digest(seed, label, _query_id(rows[index])),
                _query_id(rows[index]),
            ),
        )
        selected.extend(ordered[:target])
    return tuple(sorted(selected)), {
        "method": "deterministic_hash_downsample_equal_test_eeg_occurrences",
        "eligible_query_count_by_book": {
            book: len(by_book[book]) for book in BOOK_IDS
        },
        "selected_query_count_by_book": {book: target for book in BOOK_IDS},
    }


def _candidate_membership(
    rows: Sequence[Mapping[str, object]], candidate_ids: Sequence[str]
) -> dict[str, int]:
    books_by_id: dict[str, set[str]] = defaultdict(set)
    candidate_set = set(candidate_ids)
    for row in rows:
        text_id = str(row["span_text_id"])
        if text_id in candidate_set:
            books_by_id[text_id].add(str(row["book_id"]))
    return {
        "garnettdream_only": sum(
            books == {"garnettdream"} for books in books_by_id.values()
        ),
        "littleprince_only": sum(
            books == {"littleprince"} for books in books_by_id.values()
        ),
        "shared_between_books": sum(
            books == set(BOOK_IDS) for books in books_by_id.values()
        ),
        "absent_from_membership_rows": len(candidate_set - set(books_by_id)),
    }


def build_book_stratified_protocols(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    *,
    pool_sizes: Sequence[int] = (20, 100, 200),
    seed: int = 42,
) -> tuple[RetrievalProtocol, ...]:
    """Build mixed, balanced-mixed, and individual-book protocols."""

    sizes = tuple(sorted({int(value) for value in pool_sizes}))
    if not sizes or any(value < 2 or value % 2 for value in sizes):
        raise ValueError("Balanced pool sizes must be positive even integers")
    all_indices = tuple(range(len(test_rows)))
    semantic_indices = tuple(
        index for index, row in enumerate(test_rows) if row["is_semantic_unit"]
    )
    all_ids_by_book = {
        book: tuple(
            sorted(
                {
                    str(row["span_text_id"])
                    for row in test_rows
                    if str(row["book_id"]) == book
                }
            )
        )
        for book in BOOK_IDS
    }
    semantic_ids_by_book = {
        book: tuple(
            sorted(
                {
                    str(row["span_text_id"])
                    for row in test_rows
                    if str(row["book_id"]) == book and row["is_semantic_unit"]
                }
            )
        )
        for book in BOOK_IDS
    }
    protocols: list[RetrievalProtocol] = []

    def add(
        name: str,
        family: str,
        candidate_ids: Sequence[str],
        eligible_indices: Sequence[int],
        universe_indices: Sequence[int],
        *,
        balanced_queries: bool = False,
        candidate_selection: Mapping[str, object] | None = None,
    ) -> None:
        candidates = tuple(dict.fromkeys(str(value) for value in candidate_ids))
        candidate_set = set(candidates)
        eligible = tuple(
            int(index)
            for index in eligible_indices
            if str(test_rows[index]["span_text_id"]) in candidate_set
        )
        if balanced_queries:
            eligible_counts = Counter(
                str(test_rows[index]["book_id"]) for index in eligible
            )
            if min(eligible_counts.get(book, 0) for book in BOOK_IDS) < 1:
                queries = ()
                query_selection = {
                    "method": "deterministic_hash_downsample_equal_test_eeg_occurrences",
                    "status": "not_evaluable",
                    "reason": "at_least_one_book_has_no_eligible_test_query",
                    "eligible_query_count_by_book": {
                        book: eligible_counts.get(book, 0) for book in BOOK_IDS
                    },
                    "selected_query_count_by_book": {
                        book: 0 for book in BOOK_IDS
                    },
                }
            else:
                queries, query_selection = balance_query_indices(
                    test_rows, eligible, seed=seed, label=name
                )
        else:
            queries = eligible
            query_selection = {
                "method": "all_eligible_test_eeg_occurrences",
                "eligible_query_count_by_book": dict(
                    Counter(str(test_rows[index]["book_id"]) for index in eligible)
                ),
                "selected_query_count_by_book": dict(
                    Counter(str(test_rows[index]["book_id"]) for index in queries)
                ),
            }
        if not candidates:
            raise ValueError(f"Protocol {name} has no candidates")
        if not queries and not balanced_queries:
            query_selection = {
                **query_selection,
                "status": "not_evaluable",
                "reason": "candidate_pool_has_no_test_positive_occurrence",
            }
        protocols.append(
            RetrievalProtocol(
                name=name,
                family=family,
                candidate_ids=candidates,
                query_indices=queries,
                query_universe_indices=tuple(int(value) for value in universe_indices),
                candidate_selection={
                    **(candidate_selection or {}),
                    "candidate_count": len(candidates),
                    "membership": _candidate_membership(
                        tuple(train_rows) + tuple(test_rows), candidates
                    ),
                },
                query_selection=query_selection,
            )
        )

    mixed_all_ids = tuple(sorted(set().union(*map(set, all_ids_by_book.values()))))
    mixed_semantic_ids = tuple(
        sorted(set().union(*map(set, semantic_ids_by_book.values())))
    )
    add(
        "mixed_all_full",
        "full_all_windows",
        mixed_all_ids,
        all_indices,
        all_indices,
        candidate_selection={"method": "all_unique_test_texts_both_books"},
    )
    add(
        "mixed_semantic_full",
        "full_semantic_windows",
        mixed_semantic_ids,
        semantic_indices,
        semantic_indices,
        candidate_selection={"method": "all_unique_test_semantic_texts_both_books"},
    )
    for book in BOOK_IDS:
        book_all_indices = tuple(
            index
            for index, row in enumerate(test_rows)
            if str(row["book_id"]) == book
        )
        book_semantic_indices = tuple(
            index for index in book_all_indices if test_rows[index]["is_semantic_unit"]
        )
        add(
            f"{book}_all_full",
            "full_all_windows",
            all_ids_by_book[book],
            book_all_indices,
            book_all_indices,
            candidate_selection={"method": "all_unique_test_texts_one_book", "book_id": book},
        )
        add(
            f"{book}_semantic_full",
            "full_semantic_windows",
            semantic_ids_by_book[book],
            book_semantic_indices,
            book_semantic_indices,
            candidate_selection={
                "method": "all_unique_test_semantic_texts_one_book",
                "book_id": book,
            },
        )

    balanced_all_quota = min(len(values) for values in all_ids_by_book.values())
    balanced_all, all_contributed = balanced_unique_candidates(
        all_ids_by_book,
        quota_per_book=balanced_all_quota,
        seed=seed,
        label="balanced_all_full_candidates",
        preserve_input_order=False,
    )
    add(
        "balanced_mixed_all_full",
        "full_all_windows",
        balanced_all,
        all_indices,
        all_indices,
        balanced_queries=True,
        candidate_selection={
            "method": "equal_unique_test_vocabulary_contribution_per_book",
            "quota_per_book": balanced_all_quota,
            "contributed_candidate_count_by_book": {
                book: len(all_contributed[book]) for book in BOOK_IDS
            },
        },
    )
    balanced_semantic_quota = min(
        len(values) for values in semantic_ids_by_book.values()
    )
    balanced_semantic, semantic_contributed = balanced_unique_candidates(
        semantic_ids_by_book,
        quota_per_book=balanced_semantic_quota,
        seed=seed,
        label="balanced_semantic_full_candidates",
        preserve_input_order=False,
    )
    add(
        "balanced_mixed_semantic_full",
        "full_semantic_windows",
        balanced_semantic,
        semantic_indices,
        semantic_indices,
        balanced_queries=True,
        candidate_selection={
            "method": "equal_unique_test_semantic_vocabulary_contribution_per_book",
            "quota_per_book": balanced_semantic_quota,
            "contributed_candidate_count_by_book": {
                book: len(semantic_contributed[book]) for book in BOOK_IDS
            },
        },
    )

    mixed_frequency = _frequency_order(train_rows)
    frequency_by_book = {
        book: _frequency_order(train_rows, book_id=book) for book in BOOK_IDS
    }
    for size in sizes:
        mixed_candidates = mixed_frequency[: min(size, len(mixed_frequency))]
        add(
            f"mixed_semantic_frequency_top{size}",
            "train_frequency_semantic",
            mixed_candidates,
            semantic_indices,
            semantic_indices,
            candidate_selection={
                "method": "combined_books_train_semantic_eeg_occurrence_frequency",
                "requested_candidate_count": size,
            },
        )
        for book in BOOK_IDS:
            book_semantic_indices = tuple(
                index
                for index in semantic_indices
                if str(test_rows[index]["book_id"]) == book
            )
            actual = min(size, len(frequency_by_book[book]))
            add(
                f"{book}_semantic_frequency_top{size}",
                "train_frequency_semantic",
                frequency_by_book[book][:actual],
                book_semantic_indices,
                book_semantic_indices,
                candidate_selection={
                    "method": "one_book_train_semantic_eeg_occurrence_frequency",
                    "book_id": book,
                    "requested_candidate_count": size,
                    "actual_candidate_count": actual,
                },
            )
        balanced_frequency, frequency_contributed = balanced_unique_candidates(
            frequency_by_book,
            quota_per_book=size // 2,
            seed=seed,
            label=f"balanced_frequency_top{size}",
            preserve_input_order=True,
        )
        add(
            f"balanced_mixed_semantic_frequency_top{size}",
            "train_frequency_semantic",
            balanced_frequency,
            semantic_indices,
            semantic_indices,
            balanced_queries=True,
            candidate_selection={
                "method": "equal_per_book_train_frequency_candidate_contribution",
                "requested_candidate_count": size,
                "quota_per_book": size // 2,
                "contributed_candidate_count_by_book": {
                    book: len(frequency_contributed[book]) for book in BOOK_IDS
                },
            },
        )
    names = [protocol.name for protocol in protocols]
    if len(names) != len(set(names)):
        raise AssertionError("Protocol names are not unique")
    return tuple(protocols)


def _single_positive_ranks(
    scores: torch.Tensor, positive_indices: torch.Tensor
) -> list[int]:
    rows = torch.arange(scores.shape[0], device=scores.device)
    positive_scores = scores[rows, positive_indices]
    return (scores >= positive_scores.unsqueeze(1)).sum(dim=1).cpu().tolist()


def _macro_metrics(ranks: Sequence[int], text_ids: Sequence[str]) -> RetrievalMetrics:
    grouped: dict[str, list[int]] = defaultdict(list)
    for rank, text_id in zip(ranks, text_ids, strict=True):
        grouped[text_id].append(int(rank))
    records = [metrics_from_ranks(values) for values in grouped.values()]
    return RetrievalMetrics(
        recall_at_1=sum(item.recall_at_1 for item in records) / len(records),
        recall_at_5=sum(item.recall_at_5 for item in records) / len(records),
        recall_at_10=sum(item.recall_at_10 for item in records) / len(records),
        median_rank=sum(item.median_rank for item in records) / len(records),
        mean_reciprocal_rank=(
            sum(item.mean_reciprocal_rank for item in records) / len(records)
        ),
    )


def _shortcut_metrics(
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    protocol: RetrievalProtocol,
    *,
    seed: int,
) -> dict[str, RetrievalMetrics]:
    candidates = protocol.candidate_ids
    candidate_set = set(candidates)
    queries = [test_rows[index] for index in protocol.query_indices]
    global_counts = Counter(
        str(row["span_text_id"])
        for row in train_rows
        if str(row["span_text_id"]) in candidate_set
    )
    global_order = sorted(candidates, key=lambda value: (-global_counts[value], value))
    global_rank = {value: rank for rank, value in enumerate(global_order, 1)}
    frequency_ranks = [global_rank[str(row["span_text_id"])] for row in queries]

    book_counts = Counter(
        (str(row["book_id"]), str(row["span_text_id"]))
        for row in train_rows
        if str(row["span_text_id"]) in candidate_set
    )
    book_rank_maps = {}
    for book in BOOK_IDS:
        order = sorted(
            candidates, key=lambda value: (-book_counts[(book, value)], value)
        )
        book_rank_maps[book] = {value: rank for rank, value in enumerate(order, 1)}
    book_ranks = [
        book_rank_maps[str(row["book_id"])][str(row["span_text_id"])]
        for row in queries
    ]

    subject_counts = Counter(
        (str(row["subject_group_id"]), str(row["span_text_id"]))
        for row in train_rows
        if str(row["span_text_id"]) in candidate_set
    )
    subject_rank_maps = {}
    for subject in sorted({str(row["subject_group_id"]) for row in queries}):
        order = sorted(
            candidates,
            key=lambda value: (-subject_counts[(subject, value)], value),
        )
        subject_rank_maps[subject] = {
            value: rank for rank, value in enumerate(order, 1)
        }
    subject_ranks = [
        subject_rank_maps[str(row["subject_group_id"])][str(row["span_text_id"])]
        for row in queries
    ]

    candidates_by_position: dict[int, set[str]] = defaultdict(set)
    for index in protocol.query_universe_indices:
        row = test_rows[index]
        text_id = str(row["span_text_id"])
        if text_id in candidate_set:
            candidates_by_position[int(row["stimulus_position"])].add(text_id)
    position_ranks = []
    for row in queries:
        query_id = _query_id(row)
        positive = str(row["span_text_id"])
        tied = sorted(
            candidates_by_position[int(row["stimulus_position"])],
            key=lambda value: _stable_digest(
                seed, f"position-{query_id}", value
            ),
        )
        position_ranks.append(tied.index(positive) + 1)
    random = expected_random_retrieval_metrics(len(candidates))
    return {
        "random_analytical": random,
        "duration_only_fixed_length_tie": random,
        "train_frequency": metrics_from_ranks(frequency_ranks),
        "book_id_train_frequency": metrics_from_ranks(book_ranks),
        "subject_id_train_frequency": metrics_from_ranks(subject_ranks),
        "sentence_position_test_occurrence_oracle": metrics_from_ranks(
            position_ranks
        ),
    }


@torch.inference_mode()
def evaluate_book_stratified_protocols(
    model: torch.nn.Module,
    dataset: object,
    train_rows: Sequence[Mapping[str, object]],
    test_rows: Sequence[Mapping[str, object]],
    protocols: Sequence[RetrievalProtocol],
    *,
    text_target_provider: Callable[[Mapping[str, object]], np.ndarray],
    representative_by_text_id: Mapping[str, Mapping[str, object]],
    device: torch.device,
    query_batch_size: int = 64,
    candidate_batch_size: int = 4096,
    seed: int = 42,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Encode every test EEG once and score all stratified protocols."""

    union_ids = tuple(
        sorted({text_id for protocol in protocols for text_id in protocol.candidate_ids})
    )
    union_index = {text_id: index for index, text_id in enumerate(union_ids)}
    states = torch.from_numpy(
        np.stack(
            [
                np.ascontiguousarray(
                    text_target_provider(representative_by_text_id[text_id])
                )
                for text_id in union_ids
            ]
        )
    )
    if device.type == "cuda":
        states = states.to(device)
    query_sets = {
        protocol.name: set(protocol.query_indices) for protocol in protocols
    }
    candidate_columns = {
        protocol.name: torch.as_tensor(
            [union_index[text_id] for text_id in protocol.candidate_ids],
            device=device,
        )
        for protocol in protocols
    }
    positive_maps = {
        protocol.name: {
            text_id: index for index, text_id in enumerate(protocol.candidate_ids)
        }
        for protocol in protocols
    }
    ranks: dict[str, list[int]] = {protocol.name: [] for protocol in protocols}
    text_ids: dict[str, list[str]] = {protocol.name: [] for protocol in protocols}
    query_ids: dict[str, list[str]] = {protocol.name: [] for protocol in protocols}
    books: dict[str, list[str]] = {protocol.name: [] for protocol in protocols}
    loader = DataLoader(
        dataset,
        batch_size=query_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fixed_character_spans,
    )
    model.eval()
    offset = 0
    for batch in loader:
        metadata = batch["metadata"]
        estimates = model.encode_eeg(
            batch["eeg"].to(device, non_blocking=True),
            batch["subject_index"].to(device, non_blocking=True),
        )
        score_chunks = [
            model.get_compressed_text_scores(
                estimates, states[start : start + candidate_batch_size]
            )
            for start in range(0, len(union_ids), candidate_batch_size)
        ]
        union_scores = torch.cat(score_chunks, dim=1)
        batch_global_indices = tuple(range(offset, offset + len(metadata)))
        for protocol in protocols:
            local_rows = [
                local
                for local, global_index in enumerate(batch_global_indices)
                if global_index in query_sets[protocol.name]
            ]
            if not local_rows:
                continue
            local_tensor = torch.as_tensor(local_rows, device=device)
            protocol_scores = union_scores.index_select(0, local_tensor).index_select(
                1, candidate_columns[protocol.name]
            )
            positives = torch.as_tensor(
                [
                    positive_maps[protocol.name][
                        str(metadata[local]["span_text_id"])
                    ]
                    for local in local_rows
                ],
                device=device,
            )
            ranks[protocol.name].extend(
                _single_positive_ranks(protocol_scores, positives)
            )
            text_ids[protocol.name].extend(
                str(metadata[local]["span_text_id"]) for local in local_rows
            )
            query_ids[protocol.name].extend(
                _query_id(metadata[local]) for local in local_rows
            )
            books[protocol.name].extend(
                str(metadata[local]["book_id"]) for local in local_rows
            )
        offset += len(metadata)
    if offset != len(dataset):
        raise ValueError("Test DataLoader did not traverse the complete dataset")

    report: dict[str, object] = {}
    rank_rows: list[dict[str, object]] = []
    for protocol in protocols:
        name = protocol.name
        if len(ranks[name]) != len(protocol.query_indices):
            raise ValueError(f"Protocol {name} did not emit one rank per query")
        if not ranks[name]:
            report[name] = {
                "status": "not_evaluable",
                "family": protocol.family,
                "candidate_count": len(protocol.candidate_ids),
                "query_count": 0,
                "represented_positive_class_count": 0,
                "candidate_ids": list(protocol.candidate_ids),
                "candidate_selection": dict(protocol.candidate_selection),
                "query_selection": dict(protocol.query_selection),
                "micro_metrics": None,
                "macro_class_balanced_metrics": None,
                "metrics_by_book": {},
                "shortcut_baselines": None,
            }
            continue
        micro = metrics_from_ranks(ranks[name])
        macro = _macro_metrics(ranks[name], text_ids[name])
        by_book_metrics = {}
        for book in BOOK_IDS:
            local = [
                rank for rank, value in zip(ranks[name], books[name], strict=True)
                if value == book
            ]
            if local:
                by_book_metrics[book] = asdict(metrics_from_ranks(local))
        shortcuts = _shortcut_metrics(
            train_rows, test_rows, protocol, seed=seed
        )
        report[name] = {
            "status": "evaluated",
            "family": protocol.family,
            "candidate_count": len(protocol.candidate_ids),
            "query_count": len(ranks[name]),
            "represented_positive_class_count": len(set(text_ids[name])),
            "candidate_ids": list(protocol.candidate_ids),
            "candidate_selection": dict(protocol.candidate_selection),
            "query_selection": dict(protocol.query_selection),
            "micro_metrics": asdict(micro),
            "macro_class_balanced_metrics": asdict(macro),
            "metrics_by_book": by_book_metrics,
            "shortcut_baselines": {
                key: asdict(value) for key, value in shortcuts.items()
            },
        }
        rank_rows.extend(
            {
                "protocol": name,
                "query_id": query_id,
                "span_text_id": text_id,
                "book_id": book,
                "rank": rank,
            }
            for query_id, text_id, book, rank in zip(
                query_ids[name], text_ids[name], books[name], ranks[name], strict=True
            )
        )
    return report, rank_rows
