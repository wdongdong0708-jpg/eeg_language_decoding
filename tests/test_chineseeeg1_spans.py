from collections import Counter

from data.chineseeeg1_spans import CharacterSpanSpec, iter_chineseeeg1_character_spans


def _row() -> dict[str, object]:
    return {
        "dataset_version": "ChineseEEG1",
        "paradigm": "silent_reading",
        "subject_id": "01",
        "session_id": "LittlePrince",
        "book_id": "littleprince",
        "chapter_id": "1",
        "sentence_id": "sentence-1",
        "global_text_id": "global-1",
        "raw_text": "甲乙，丙丁戊己庚辛壬癸",
        "eeg_file": "fake.eeg",
        "eeg_start_sample": 100,
        "eeg_end_sample": 1175,
        "eeg_sampling_rate": 256.0,
        "quality_flag": "ok",
        "split_group_id": "group-1",
        "record_id": "record-1",
        "run_id": "01",
        "block_id": "block-1",
        "content_id": "content-1",
        "stimulus_position": 1,
        "text_alignment_status": "exact",
    }


def _audit() -> dict[str, object]:
    return {
        "verdict": "verified_approximate_only",
        "allowed_timeline_methods": ["event_affine", "fixed_dwell_sensitivity"],
    }


def test_fixed_spans_inherit_split_and_never_pad() -> None:
    counters: Counter[str] = Counter()
    spans = list(
        iter_chineseeeg1_character_spans(
            [_row()],
            record_partitions={"record-1": "test"},
            timeline_audit=_audit(),
            spec=CharacterSpanSpec(
                span_lengths=(4, 6, 8),
                neural_delay_ms=0.0,
            ),
            counters=counters,
        )
    )
    assert len(spans) == 15
    assert {span.split for span in spans} == {"test"}
    assert {span.padding_samples for span in spans} == {0}
    assert {span.exposes_padding_mask for span in spans} == {False}
    assert {
        length: {span.model_eeg_sample_count for span in spans if span.span_char_count == length}
        for length in (4, 6, 8)
    } == {4: {358}, 6: {538}, 8: {717}}
    first = spans[0]
    assert first.span_text == "甲乙丙丁"
    assert first.span_surface_text == "甲乙，丙丁"
    assert (first.span_start_char, first.span_end_char) == (0, 5)


def test_delay_drops_spans_that_would_cross_the_source_row() -> None:
    baseline = list(
        iter_chineseeeg1_character_spans(
            [_row()],
            record_partitions={"record-1": "train"},
            timeline_audit=_audit(),
            spec=CharacterSpanSpec(span_lengths=(4,), neural_delay_ms=0.0),
        )
    )
    delayed = list(
        iter_chineseeeg1_character_spans(
            [_row()],
            record_partitions={"record-1": "train"},
            timeline_audit=_audit(),
            spec=CharacterSpanSpec(span_lengths=(4,), neural_delay_ms=200.0),
        )
    )
    assert len(delayed) < len(baseline)
    assert all(span.eeg_stop_sample <= span.source_row_stop_sample for span in delayed)


def test_clean_k4_fixed_dwell_uses_nonoverlap_and_fixed_source_samples() -> None:
    spans = list(
        iter_chineseeeg1_character_spans(
            [_row()],
            record_partitions={"record-1": "train"},
            timeline_audit=_audit(),
            spec=CharacterSpanSpec(
                span_lengths=(4,),
                stride_characters=4,
                timeline_method="fixed_dwell_sensitivity",
                neural_delay_ms=0.0,
                left_context_ms=200.0,
                right_context_ms=600.0,
            ),
        )
    )

    # The first block is dropped because its left context would cross ROWS;
    # the next complete non-overlapping block is retained without padding.
    assert len(spans) == 1
    assert spans[0].span_start_clock == 4
    assert spans[0].span_end_clock == 8
    assert spans[0].source_eeg_sample_count == 563
    assert spans[0].model_eeg_sample_count == 563
    assert spans[0].padding_samples == 0


def test_allowed_clock_start_keeps_only_the_first_complete_block() -> None:
    spans = list(
        iter_chineseeeg1_character_spans(
            [_row()],
            record_partitions={"record-1": "validation"},
            timeline_audit=_audit(),
            spec=CharacterSpanSpec(
                span_lengths=(4,),
                stride_characters=4,
                allowed_clock_starts=(0,),
                timeline_method="fixed_dwell_sensitivity",
                neural_delay_ms=0.0,
            ),
        )
    )

    assert len(spans) == 1
    assert (spans[0].span_start_clock, spans[0].span_end_clock) == (0, 4)
    assert spans[0].source_eeg_sample_count == 358
    assert spans[0].model_eeg_sample_count == 358
