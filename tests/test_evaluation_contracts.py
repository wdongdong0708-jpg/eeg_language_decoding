import pytest

from evaluation.candidate_pools import (
    CandidateMetadata,
    LengthMatchPolicy,
    length_matched_candidates,
)
from evaluation.retrieval_metrics import metrics_from_ranks
from evaluation.shortcut_baselines import (
    REQUIRED_SHORTCUT_BASELINES,
    validate_shortcut_baseline_names,
)


def test_rank_metrics_use_one_based_ranks() -> None:
    metrics = metrics_from_ranks([1, 2, 10, 20])
    assert metrics.recall_at_1 == 0.25
    assert metrics.recall_at_5 == 0.5
    assert metrics.recall_at_10 == 0.75


def test_length_pool_never_silently_relaxes_minimum() -> None:
    query = CandidateMetadata("positive", character_count=10, duration_sec=2.0)
    with pytest.raises(ValueError, match="minimum"):
        length_matched_candidates(
            query,
            [
                query,
                CandidateMetadata("negative", character_count=10, duration_sec=2.1),
            ],
            policy=LengthMatchPolicy(minimum_pool_size=3),
        )


def test_all_shortcut_baselines_are_required() -> None:
    validate_shortcut_baseline_names(set(REQUIRED_SHORTCUT_BASELINES))
    with pytest.raises(ValueError, match="duration_only"):
        validate_shortcut_baseline_names({"random"})

