import torch

from evaluation.speech_retrieval import (
    position_local_candidate_pools,
    ranks_from_candidate_pools,
    ranks_from_score_matrix,
)
from training.samplers import UniqueTargetBatchSampler


def test_unique_target_sampler_never_places_duplicates_in_batch() -> None:
    target_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    sampler = UniqueTargetBatchSampler(
        target_ids,
        batch_size=3,
        seed=42,
        drop_last=False,
    )
    visited: list[int] = []
    for batch in sampler:
        visited.extend(batch)
        batch_targets = [target_ids[index] for index in batch]
        assert len(batch_targets) == len(set(batch_targets))
    assert sorted(visited) == list(range(len(target_ids)))


def test_sampler_is_deterministic_by_epoch() -> None:
    target_ids = [str(index // 2) for index in range(20)]
    sampler = UniqueTargetBatchSampler(
        target_ids,
        batch_size=4,
        seed=42,
    )
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_retrieval_ranks_use_pessimistic_ties() -> None:
    scores = torch.tensor([[3.0, 3.0, 1.0], [1.0, 2.0, 3.0]])
    assert ranks_from_score_matrix(scores, [0, 2]) == [2, 1]
    assert ranks_from_score_matrix(
        scores,
        [0, 2],
        tie_policy="optimistic",
    ) == [1, 1]


def test_position_local_pool_is_fixed_size_and_contains_positive() -> None:
    positions = [{0}, {10}, {20}, {30}, {40}]
    pools = position_local_candidate_pools(
        [18],
        positions,
        [2],
        pool_size=3,
    )
    assert len(pools[0]) == 3
    assert 2 in pools[0]
    scores = torch.tensor([[0.0, 0.5, 1.0, 0.8, 0.1]])
    assert ranks_from_candidate_pools(scores, [2], pools) == [1]
