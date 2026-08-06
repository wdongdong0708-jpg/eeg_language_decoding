"""Class-balanced validation retrieval for static ChineseEEG1 text targets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.chineseeeg1_span_dataset import collate_fixed_character_spans
from evaluation.retrieval_metrics import RetrievalMetrics, metrics_from_ranks


@dataclass(frozen=True, slots=True)
class BalancedTextRetrievalResult:
    candidate_count: int
    query_count: int
    class_count: int
    micro: RetrievalMetrics
    macro: RetrievalMetrics
    ranks: tuple[int, ...]
    target_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    query_ids: tuple[str, ...]


def macro_metrics_from_ranks(
    ranks: Sequence[int], target_ids: Sequence[str]
) -> RetrievalMetrics:
    if len(ranks) != len(target_ids) or not ranks:
        raise ValueError("ranks and non-empty target_ids must align")
    grouped: dict[str, list[int]] = defaultdict(list)
    for rank, target_id in zip(ranks, target_ids, strict=True):
        grouped[str(target_id)].append(int(rank))
    records = [metrics_from_ranks(grouped[key]) for key in sorted(grouped)]
    return RetrievalMetrics(
        recall_at_1=sum(record.recall_at_1 for record in records) / len(records),
        recall_at_5=sum(record.recall_at_5 for record in records) / len(records),
        recall_at_10=sum(record.recall_at_10 for record in records) / len(records),
        median_rank=sum(record.median_rank for record in records) / len(records),
        mean_reciprocal_rank=(
            sum(record.mean_reciprocal_rank for record in records) / len(records)
        ),
    )


def _unique_candidate_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], dict[str, int]]:
    by_id: dict[str, Mapping[str, object]] = {}
    text_by_id: dict[str, str] = {}
    for row in rows:
        target_id = str(row["span_text_id"])
        text = str(row["span_text"])
        previous = text_by_id.setdefault(target_id, text)
        if previous != text:
            raise ValueError("One span_text_id maps to multiple text surfaces")
        by_id.setdefault(target_id, row)
    ordered_ids = sorted(by_id)
    return [by_id[key] for key in ordered_ids], {
        key: index for index, key in enumerate(ordered_ids)
    }


@torch.inference_mode()
def evaluate_balanced_text_retrieval(
    model: torch.nn.Module,
    dataset: object,
    *,
    text_target_provider: Callable[[Mapping[str, object]], np.ndarray],
    device: torch.device,
    query_batch_size: int = 128,
    candidate_batch_size: int = 2048,
    max_queries: int | None = None,
) -> BalancedTextRetrievalResult:
    """Rank every validation occurrence against all unique validation texts."""

    if min(query_batch_size, candidate_batch_size) <= 0:
        raise ValueError("Evaluation batch sizes must be positive")
    rows = dataset.table.to_pylist()
    candidate_rows, candidate_index = _unique_candidate_rows(rows)
    candidate_embeddings: list[torch.Tensor] = []
    for start in range(0, len(candidate_rows), candidate_batch_size):
        targets = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(text_target_provider(row), dtype=np.float32)
                )
                for row in candidate_rows[start : start + candidate_batch_size]
            ]
        ).to(device)
        candidate_embeddings.append(model.encode_text(targets).cpu())
    encoded_candidates = torch.cat(candidate_embeddings, dim=0)

    loader = DataLoader(
        dataset,
        batch_size=query_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fixed_character_spans,
    )
    model.eval()
    ranks: list[int] = []
    target_ids: list[str] = []
    subject_ids: list[str] = []
    query_ids: list[str] = []
    for batch in loader:
        if max_queries is not None and len(ranks) >= max_queries:
            break
        metadata = batch["metadata"]
        eeg = batch["eeg"].to(device, non_blocking=True)
        subject_indices = batch["subject_index"].to(device, non_blocking=True)
        if max_queries is not None:
            remaining = max_queries - len(ranks)
            metadata = metadata[:remaining]
            eeg = eeg[:remaining]
            subject_indices = subject_indices[:remaining]
        estimates = model.encode_eeg(eeg, subject_indices)
        score_chunks = [
            model.objective.get_scores(
                estimates,
                encoded_candidates[start : start + candidate_batch_size].to(device),
            ).cpu()
            for start in range(0, len(candidate_rows), candidate_batch_size)
        ]
        scores = torch.cat(score_chunks, dim=1)
        positives = torch.as_tensor(
            [candidate_index[str(row["span_text_id"])] for row in metadata],
            dtype=torch.long,
        )
        positive_scores = scores.gather(1, positives[:, None])
        # Pessimistic ties: the positive itself is included and supplies rank 1.
        ranks.extend(int(value) for value in (scores >= positive_scores).sum(dim=1))
        target_ids.extend(str(row["span_text_id"]) for row in metadata)
        subject_ids.extend(str(row["subject_group_id"]) for row in metadata)
        query_ids.extend(
            f"{row['record_id']}::{row['span_event_id']}" for row in metadata
        )
    if not ranks:
        raise ValueError("Balanced retrieval evaluation produced no queries")
    return BalancedTextRetrievalResult(
        candidate_count=len(candidate_rows),
        query_count=len(ranks),
        class_count=len(set(target_ids)),
        micro=metrics_from_ranks(ranks),
        macro=macro_metrics_from_ranks(ranks, target_ids),
        ranks=tuple(ranks),
        target_ids=tuple(target_ids),
        subject_ids=tuple(subject_ids),
        query_ids=tuple(query_ids),
    )
