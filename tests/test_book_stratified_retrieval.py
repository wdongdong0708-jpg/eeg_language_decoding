from evaluation.book_stratified_retrieval import (
    balance_query_indices,
    balanced_unique_candidates,
    build_book_stratified_protocols,
)


def test_balanced_candidates_are_equal_and_unique() -> None:
    pool, contributed = balanced_unique_candidates(
        {
            "garnettdream": ["shared", "g1", "g2", "g3"],
            "littleprince": ["shared", "l1", "l2"],
        },
        quota_per_book=2,
        seed=42,
        label="test",
        preserve_input_order=True,
    )
    assert len(pool) == len(set(pool)) == 4
    assert {book: len(values) for book, values in contributed.items()} == {
        "garnettdream": 2,
        "littleprince": 2,
    }


def test_balanced_query_sampling_is_equal_and_deterministic() -> None:
    rows = []
    for book, count in (("garnettdream", 5), ("littleprince", 3)):
        for index in range(count):
            rows.append(
                {
                    "book_id": book,
                    "record_id": f"{book}-{index}",
                    "span_event_id": f"event-{index}",
                    "span_start_clock": index,
                }
            )
    first, report = balance_query_indices(
        rows, range(len(rows)), seed=42, label="test"
    )
    second, _ = balance_query_indices(
        rows, range(len(rows)), seed=42, label="test"
    )
    assert first == second
    assert len(first) == 6
    assert report["selected_query_count_by_book"] == {
        "garnettdream": 3,
        "littleprince": 3,
    }


def test_zero_positive_frequency_protocol_is_marked_not_evaluable() -> None:
    def row(book: str, text_id: str, index: int, split: str) -> dict[str, object]:
        return {
            "record_id": f"{split}-{book}-{index}",
            "span_event_id": f"event-{split}-{book}-{index}",
            "span_start_clock": index,
            "span_text_id": text_id,
            "span_text": text_id,
            "span_char_count": 2,
            "book_id": book,
            "subject_group_id": f"subject-{book}",
            "stimulus_position": index,
            "is_semantic_unit": True,
        }

    train = []
    test = []
    for book in ("garnettdream", "littleprince"):
        for index in range(110):
            train.append(row(book, f"train-{book}-{index}", index, "train"))
        test.append(row(book, f"test-{book}", 0, "test"))
    protocols = build_book_stratified_protocols(train, test, pool_sizes=(20, 100))
    target = next(
        protocol
        for protocol in protocols
        if protocol.name == "garnettdream_semantic_frequency_top20"
    )
    assert target.query_indices == ()
    assert target.query_selection["status"] == "not_evaluable"
