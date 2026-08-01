"""Retrieval evaluation for EEG queries and unique speech-window candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from evaluation.retrieval_metrics import RetrievalMetrics, metrics_from_ranks
from models.losses import NormKind, sequence_similarity

TiePolicy = Literal["optimistic", "pessimistic"]


def ranks_from_score_matrix(
    scores: torch.Tensor,
    positive_indices: Sequence[int] | torch.Tensor,
    *,
    tie_policy: TiePolicy = "pessimistic",
) -> list[int]:
    """Return one-based ranks with an explicit tie policy."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [queries, candidates]")
    positives = torch.as_tensor(
        positive_indices,
        dtype=torch.long,
        device=scores.device,
    )
    if positives.shape != (scores.shape[0],):
        raise ValueError("positive_indices must have one value per query")
    if torch.any(positives < 0) or torch.any(positives >= scores.shape[1]):
        raise ValueError("positive candidate index is out of range")
    positive_scores = scores.gather(1, positives[:, None])
    if tie_policy == "optimistic":
        ranks = 1 + (scores > positive_scores).sum(dim=1)
    elif tie_policy == "pessimistic":
        ranks = (scores >= positive_scores).sum(dim=1)
    else:
        raise ValueError(f"Unknown tie_policy: {tie_policy}")
    return [int(rank) for rank in ranks.cpu().tolist()]


def position_local_candidate_pools(
    query_positions: Sequence[int],
    candidate_position_sets: Sequence[set[int]],
    positive_indices: Sequence[int],
    *,
    pool_size: int,
) -> list[list[int]]:
    """Select the positive plus nearest-position negatives with fixed pool size."""

    if pool_size < 2 or pool_size > len(candidate_position_sets):
        raise ValueError("pool_size must be between 2 and candidate count")
    if len(query_positions) != len(positive_indices):
        raise ValueError("query_positions and positive_indices must align")
    pools: list[list[int]] = []
    for query_position, positive in zip(
        query_positions,
        positive_indices,
        strict=True,
    ):
        if not 0 <= positive < len(candidate_position_sets):
            raise ValueError("positive candidate index is out of range")
        negatives = [
            (
                min(abs(query_position - value) for value in positions),
                index,
            )
            for index, positions in enumerate(candidate_position_sets)
            if index != positive and positions
        ]
        negatives.sort()
        pool = sorted(
            [positive, *(index for _, index in negatives[: pool_size - 1])]
        )
        if len(pool) != pool_size:
            raise ValueError("Could not construct requested position-local pool")
        pools.append(pool)
    return pools


def ranks_from_candidate_pools(
    scores: torch.Tensor,
    positive_indices: Sequence[int],
    candidate_pools: Sequence[Sequence[int]],
    *,
    tie_policy: TiePolicy = "pessimistic",
) -> list[int]:
    if len(candidate_pools) != scores.shape[0]:
        raise ValueError("candidate_pools must have one pool per query")
    ranks: list[int] = []
    for query_index, (positive, pool) in enumerate(
        zip(positive_indices, candidate_pools, strict=True)
    ):
        if positive not in pool:
            raise ValueError("candidate pool does not contain its positive")
        local_positive = list(pool).index(positive)
        ranks.extend(
            ranks_from_score_matrix(
                scores[query_index : query_index + 1, list(pool)],
                [local_positive],
                tie_policy=tie_policy,
            )
        )
    return ranks


@torch.inference_mode()
def evaluate_speech_retrieval(
    model: torch.nn.Module,
    dataset: object,
    *,
    device: torch.device,
    query_batch_size: int = 4,
    candidate_batch_size: int = 8,
    norm_kind: NormKind = "y",
    max_queries: int | None = None,
    position_pool_size: int | None = None,
) -> tuple[
    dict[str, RetrievalMetrics],
    dict[str, list[int]],
    int,
]:
    """Rank each EEG query against all unique speech targets in its partition."""

    from torch.utils.data import DataLoader

    candidate_ids = sorted(set(dataset.audio_target_ids))
    candidate_index = {
        target_id: index for index, target_id in enumerate(candidate_ids)
    }
    candidate_position_sets = [
        {
            window.stimulus_position
            for window in dataset.windows
            if window.audio_target_id == target_id
        }
        for target_id in candidate_ids
    ]
    loader = DataLoader(
        dataset,
        batch_size=query_batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    all_ranks: list[int] = []
    local_ranks: list[int] = []
    query_count = 0
    for batch in loader:
        if max_queries is not None and query_count >= max_queries:
            break
        eeg = batch["eeg"].to(device)
        subject_indices = batch["subject_index"].to(device)
        if max_queries is not None:
            remaining = max_queries - query_count
            eeg = eeg[:remaining]
            subject_indices = subject_indices[:remaining]
            batch["audio_target_id"] = batch["audio_target_id"][:remaining]
            batch["stimulus_position"] = batch["stimulus_position"][:remaining]
        estimates = model(eeg, subject_indices)
        score_chunks: list[torch.Tensor] = []
        for start in range(0, len(candidate_ids), candidate_batch_size):
            chunk_ids = candidate_ids[start : start + candidate_batch_size]
            candidates = torch.stack(
                [dataset.load_speech_target(target_id) for target_id in chunk_ids]
            ).to(device)
            score_chunks.append(
                sequence_similarity(
                    estimates,
                    candidates,
                    norm_kind=norm_kind,
                ).cpu()
            )
        scores = torch.cat(score_chunks, dim=1)
        positives = [
            candidate_index[target_id] for target_id in batch["audio_target_id"]
        ]
        all_ranks.extend(
            ranks_from_score_matrix(
                scores,
                positives,
                tie_policy="pessimistic",
            )
        )
        if position_pool_size is not None:
            pools = position_local_candidate_pools(
                [int(value) for value in batch["stimulus_position"]],
                candidate_position_sets,
                positives,
                pool_size=position_pool_size,
            )
            local_ranks.extend(
                ranks_from_candidate_pools(
                    scores,
                    positives,
                    pools,
                    tie_policy="pessimistic",
                )
            )
        query_count += len(positives)
    if not all_ranks:
        raise ValueError("Retrieval evaluation produced no query ranks")
    metrics = {"global": metrics_from_ranks(all_ranks)}
    ranks = {"global": all_ranks}
    if position_pool_size is not None:
        metrics["position_local"] = metrics_from_ranks(local_ranks)
        ranks["position_local"] = local_ranks
    return metrics, ranks, len(candidate_ids)
