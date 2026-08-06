import numpy as np

from training.text_negatives import (
    NegativePolicyConfig,
    TextCandidate,
    build_pairwise_negative_policy,
    classify_candidate_pair,
    mine_typed_negatives,
)


def _candidate(
    candidate_id: str,
    text: str,
    *,
    event: str,
    text_id: str,
    global_id: str,
    start: int,
    position: int,
) -> TextCandidate:
    return TextCandidate(
        candidate_id=candidate_id,
        span_event_id=event,
        span_text_id=text_id,
        global_text_id=global_id,
        span_text=text,
        span_char_count=len(text),
        span_start_clock=start,
        span_end_clock=start + len(text),
        book_id="book",
        stimulus_position=position,
        semantic_embedding=np.asarray([1.0, 0.0]),
    )


def test_overlapping_and_exact_text_candidates_are_not_plain_negatives() -> None:
    anchor = _candidate(
        "a", "甲乙丙丁", event="event-a", text_id="text-a", global_id="g", start=0, position=10
    )
    overlap = _candidate(
        "b", "乙丙丁戊", event="event-b", text_id="text-b", global_id="g", start=1, position=10
    )
    exact = _candidate(
        "c", "甲乙丙丁", event="event-c", text_id="text-a", global_id="other", start=0, position=30
    )
    assert classify_candidate_pair(anchor, overlap).primary_type == "overlapping_local_text"
    assert classify_candidate_pair(anchor, exact).primary_type == "exact_text_equivalent"
    policy = build_pairwise_negative_policy([anchor, overlap, exact])
    assert policy.candidate_mask[0, 1] == np.False_
    assert policy.positive_weights[0, 2] == 1.0


def test_typed_negative_mining_is_deterministic_and_skips_false_negatives() -> None:
    anchor = _candidate(
        "a", "甲乙丙丁", event="event-a", text_id="text-a", global_id="g", start=0, position=10
    )
    candidates = [
        _candidate(
            f"c{i}", "戊己庚辛", event=f"e{i}", text_id=f"t{i}", global_id=f"g{i}", start=0, position=30 + i
        )
        for i in range(5)
    ]
    quotas = {"semantic_similar_different_event": 3}
    first = mine_typed_negatives(anchor, candidates, quotas=quotas, seed=42)
    second = mine_typed_negatives(anchor, candidates, quotas=quotas, seed=42)
    assert first == second
    assert len(first["semantic_similar_different_event"]) == 3
