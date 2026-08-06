"""Full-vocabulary evaluation for seen-text character sliding windows."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.chineseeeg1_span_dataset import collate_fixed_character_spans
from evaluation.retrieval_metrics import RetrievalMetrics, metrics_from_ranks


@dataclass(frozen=True, slots=True)
class SeenTextRetrievalResult:
    all_metrics: RetrievalMetrics
    all_ranks: tuple[int, ...]
    semantic_query_all_candidate_metrics: RetrievalMetrics
    semantic_query_all_candidate_ranks: tuple[int, ...]
    semantic_pool_metrics: RetrievalMetrics
    semantic_pool_ranks: tuple[int, ...]
    diagnostic_metrics: Mapping[int, RetrievalMetrics]
    diagnostic_ranks: Mapping[int, tuple[int, ...]]
    query_ids: tuple[str, ...]
    query_text_ids: tuple[str, ...]
    query_subject_ids: tuple[str, ...]
    query_is_semantic: tuple[bool, ...]
    candidate_text_ids: tuple[str, ...]
    semantic_candidate_text_ids: tuple[str, ...]


def diagnostic_rank_from_full_rank(
    *,
    full_rank: int,
    full_candidate_count: int,
    requested_pool_size: int,
    seed: int,
    query_id: str,
) -> int:
    """Sample the exact random-subpool rank via a hypergeometric draw."""

    if not 1 <= full_rank <= full_candidate_count:
        raise ValueError("Full rank is outside the candidate vocabulary")
    pool_size = min(int(requested_pool_size), full_candidate_count)
    if pool_size < 1:
        raise ValueError("Diagnostic pool size must be positive")
    if pool_size == full_candidate_count:
        return full_rank
    payload = f"{seed}\0{query_id}\0{requested_pool_size}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    generator = np.random.default_rng(local_seed)
    higher_or_tied_negatives = full_rank - 1
    lower_negatives = full_candidate_count - full_rank
    selected_higher = generator.hypergeometric(
        higher_or_tied_negatives,
        lower_negatives,
        pool_size - 1,
    )
    return 1 + int(selected_higher)


def _candidate_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], tuple[str, ...], tuple[bool, ...]]:
    representative: dict[str, Mapping[str, object]] = {}
    semantic_by_text: dict[str, bool] = {}
    surface_by_text: dict[str, str] = {}
    for row in rows:
        text_id = str(row["span_text_id"])
        surface = str(row["span_text"])
        if text_id in surface_by_text and surface_by_text[text_id] != surface:
            raise ValueError("One span_text_id maps to multiple standardized texts")
        surface_by_text[text_id] = surface
        representative.setdefault(text_id, row)
        semantic_by_text[text_id] = semantic_by_text.get(text_id, False) or bool(
            row.get("is_semantic_unit", False)
        )
    text_ids = tuple(sorted(representative))
    return (
        [representative[text_id] for text_id in text_ids],
        text_ids,
        tuple(semantic_by_text[text_id] for text_id in text_ids),
    )


def _single_positive_ranks(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
) -> list[int]:
    if scores.ndim != 2 or positive_indices.shape != (scores.shape[0],):
        raise ValueError("Scores and positive indices are inconsistent")
    row_indices = torch.arange(scores.shape[0], device=scores.device)
    positive_scores = scores[row_indices, positive_indices]
    return (
        1
        + (scores >= positive_scores.unsqueeze(1)).sum(dim=1)
        - 1
    ).to(dtype=torch.int64).cpu().tolist()


@torch.inference_mode()
def evaluate_seen_text_retrieval(
    model: torch.nn.Module,
    dataset: object,
    *,
    text_target_provider: Callable[[Mapping[str, object]], np.ndarray],
    device: torch.device,
    query_batch_size: int = 32,
    candidate_batch_size: int = 512,
    diagnostic_pool_sizes: Sequence[int] = (20, 100, 1000),
    diagnostic_seed: int = 42,
    max_queries: int | None = None,
) -> SeenTextRetrievalResult:
    rows = dataset.table.to_pylist()
    candidate_rows, candidate_ids, candidate_semantic = _candidate_rows(rows)
    candidate_states = torch.from_numpy(
        np.stack(
            [
                np.ascontiguousarray(text_target_provider(row))
                for row in candidate_rows
            ]
        )
    )
    if device.type == "cuda":
        candidate_states = candidate_states.to(device)
    candidate_index = {
        text_id: index for index, text_id in enumerate(candidate_ids)
    }
    semantic_indices = tuple(
        index for index, value in enumerate(candidate_semantic) if value
    )
    if not semantic_indices:
        raise ValueError("The test vocabulary contains no semantic candidates")
    semantic_local_index = {
        candidate_ids[global_index]: local_index
        for local_index, global_index in enumerate(semantic_indices)
    }
    loader = DataLoader(
        dataset,
        batch_size=query_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fixed_character_spans,
    )
    model.eval()
    all_ranks: list[int] = []
    semantic_all_ranks: list[int] = []
    semantic_pool_ranks: list[int] = []
    query_ids: list[str] = []
    query_text_ids: list[str] = []
    subject_ids: list[str] = []
    semantic_flags: list[bool] = []
    for batch in loader:
        if max_queries is not None and len(all_ranks) >= max_queries:
            break
        metadata = batch["metadata"]
        eeg = batch["eeg"]
        subject_indices = batch["subject_index"]
        if max_queries is not None:
            remaining = max_queries - len(all_ranks)
            eeg = eeg[:remaining]
            subject_indices = subject_indices[:remaining]
            metadata = metadata[:remaining]
        estimates = model.encode_eeg(
            eeg.to(device, non_blocking=True),
            subject_indices.to(device, non_blocking=True),
        )
        score_chunks = [
            model.get_compressed_text_scores(
                estimates,
                candidate_states[start : start + candidate_batch_size].to(
                    device, non_blocking=True
                ),
            ).cpu()
            for start in range(0, len(candidate_ids), candidate_batch_size)
        ]
        scores = torch.cat(score_chunks, dim=1)
        positives = torch.as_tensor(
            [candidate_index[str(row["span_text_id"])] for row in metadata],
            dtype=torch.long,
        )
        batch_all_ranks = _single_positive_ranks(scores, positives)
        all_ranks.extend(batch_all_ranks)
        batch_semantic_rows = [
            index for index, row in enumerate(metadata) if row["is_semantic_unit"]
        ]
        if batch_semantic_rows:
            semantic_all_ranks.extend(
                batch_all_ranks[index] for index in batch_semantic_rows
            )
            semantic_scores = scores[
                batch_semantic_rows
            ][:, list(semantic_indices)]
            semantic_positives = torch.as_tensor(
                [
                    semantic_local_index[str(metadata[index]["span_text_id"])]
                    for index in batch_semantic_rows
                ],
                dtype=torch.long,
            )
            semantic_pool_ranks.extend(
                _single_positive_ranks(semantic_scores, semantic_positives)
            )
        for row in metadata:
            query_ids.append(
                f"{row['record_id']}::{row['span_event_id']}::{row['span_start_clock']}"
            )
            query_text_ids.append(str(row["span_text_id"]))
            subject_ids.append(str(row["subject_group_id"]))
            semantic_flags.append(bool(row["is_semantic_unit"]))
    expected = len(dataset) if max_queries is None else min(len(dataset), max_queries)
    if len(all_ranks) != expected:
        raise ValueError("Evaluation did not emit one rank per query")
    diagnostics: dict[int, tuple[int, ...]] = {}
    for size in sorted({int(value) for value in diagnostic_pool_sizes}):
        diagnostics[size] = tuple(
            diagnostic_rank_from_full_rank(
                full_rank=rank,
                full_candidate_count=len(candidate_ids),
                requested_pool_size=size,
                seed=diagnostic_seed,
                query_id=query_id,
            )
            for rank, query_id in zip(all_ranks, query_ids, strict=True)
        )
    return SeenTextRetrievalResult(
        all_metrics=metrics_from_ranks(all_ranks),
        all_ranks=tuple(all_ranks),
        semantic_query_all_candidate_metrics=metrics_from_ranks(semantic_all_ranks),
        semantic_query_all_candidate_ranks=tuple(semantic_all_ranks),
        semantic_pool_metrics=metrics_from_ranks(semantic_pool_ranks),
        semantic_pool_ranks=tuple(semantic_pool_ranks),
        diagnostic_metrics={
            size: metrics_from_ranks(ranks) for size, ranks in diagnostics.items()
        },
        diagnostic_ranks=diagnostics,
        query_ids=tuple(query_ids),
        query_text_ids=tuple(query_text_ids),
        query_subject_ids=tuple(subject_ids),
        query_is_semantic=tuple(semantic_flags),
        candidate_text_ids=candidate_ids,
        semantic_candidate_text_ids=tuple(
            candidate_ids[index] for index in semantic_indices
        ),
    )
