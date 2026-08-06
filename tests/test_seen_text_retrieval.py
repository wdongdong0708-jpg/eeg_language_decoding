from evaluation.seen_text_retrieval import diagnostic_rank_from_full_rank


def test_hypergeometric_diagnostic_rank_is_stable_and_bounded() -> None:
    first = diagnostic_rank_from_full_rank(
        full_rank=100,
        full_candidate_count=1000,
        requested_pool_size=20,
        seed=42,
        query_id="q1",
    )
    second = diagnostic_rank_from_full_rank(
        full_rank=100,
        full_candidate_count=1000,
        requested_pool_size=20,
        seed=42,
        query_id="q1",
    )
    assert first == second
    assert 1 <= first <= 20


def test_full_size_diagnostic_preserves_rank() -> None:
    assert (
        diagnostic_rank_from_full_rank(
            full_rank=71,
            full_candidate_count=100,
            requested_pool_size=1000,
            seed=1,
            query_id="q2",
        )
        == 71
    )
