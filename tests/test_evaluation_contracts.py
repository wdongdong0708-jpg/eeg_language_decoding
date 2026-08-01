import pytest

from evaluation.candidate_pools import (
    CandidateMetadata,
    LengthMatchPolicy,
    length_matched_candidates,
)
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    expected_random_retrieval_metrics,
    metrics_by_group,
    metrics_from_ranks,
)
from evaluation.shortcut_baselines import (
    REQUIRED_SHORTCUT_BASELINES,
    validate_shortcut_baseline_names,
)


def test_rank_metrics_use_one_based_ranks() -> None:
    metrics = metrics_from_ranks([1, 2, 10, 20])
    assert metrics.recall_at_1 == 0.25
    assert metrics.recall_at_5 == 0.5
    assert metrics.recall_at_10 == 0.75


def test_expected_random_metrics_match_uniform_candidate_ranks() -> None:
    metrics = expected_random_retrieval_metrics(20)
    assert metrics.recall_at_1 == pytest.approx(0.05)
    assert metrics.recall_at_5 == pytest.approx(0.25)
    assert metrics.recall_at_10 == pytest.approx(0.5)
    assert metrics.median_rank == pytest.approx(10.5)
    assert metrics.mean_reciprocal_rank == pytest.approx(
        sum(1.0 / rank for rank in range(1, 21)) / 20
    )


def test_grouped_metrics_keep_a_global_candidate_rank_scale() -> None:
    grouped = metrics_by_group(
        [1, 10, 2, 20],
        ["subject-b", "subject-b", "subject-a", "subject-a"],
    )

    assert list(grouped) == ["subject-a", "subject-b"]
    assert grouped["subject-a"].recall_at_1 == 0.0
    assert grouped["subject-a"].recall_at_10 == 0.5
    assert grouped["subject-b"].recall_at_1 == 0.5
    assert grouped["subject-b"].recall_at_10 == 1.0


def test_three_seed_aggregate_uses_sample_standard_deviation() -> None:
    aggregate = aggregate_retrieval_metrics(
        [
            metrics_from_ranks([1, 20]),
            metrics_from_ranks([1, 1]),
            metrics_from_ranks([20, 20]),
        ],
        ddof=1,
    )

    assert aggregate.count == 3
    assert aggregate.mean.recall_at_1 == pytest.approx(0.5)
    assert aggregate.std.recall_at_1 == pytest.approx(0.5)


def test_grouped_metrics_require_one_group_per_rank() -> None:
    with pytest.raises(ValueError, match="same length"):
        metrics_by_group([1, 2], ["subject-a"])


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
