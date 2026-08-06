from pathlib import Path

import numpy as np

from data.chineseeeg1_span_dataset import (
    ChineseEEG1SpanDataset,
    collate_fixed_character_spans,
)
from data.chineseeeg1_spans import (
    CharacterSpanSpec,
    iter_chineseeeg1_character_spans,
    write_character_span_parquet,
)


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
        "raw_text": "甲乙丙丁戊己庚辛壬癸",
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


def test_dataset_resamples_without_padding_or_mask(tmp_path: Path) -> None:
    spans = iter_chineseeeg1_character_spans(
        [_row()],
        record_partitions={"record-1": "test"},
        timeline_audit={"allowed_timeline_methods": ["event_affine"]},
        spec=CharacterSpanSpec(span_lengths=(4,), neural_delay_ms=0.0),
    )
    index = tmp_path / "spans.parquet"
    write_character_span_parquet(index, spans)

    def reader(_: str, start: int, stop: int) -> np.ndarray:
        return np.tile(
            np.linspace(-1.0, 1.0, stop - start, dtype=np.float32),
            (3, 1),
        )

    dataset = ChineseEEG1SpanDataset(
        index,
        partition="test",
        span_char_count=4,
        eeg_reader=reader,
        text_target_provider=lambda _: np.ones(5, dtype=np.float32),
    )
    first = dataset[0]
    second = dataset[1]
    assert first["eeg"].shape == (3, 358)
    assert "padding_mask" not in first
    batch = collate_fixed_character_spans([first, second])
    assert batch["eeg"].shape == (2, 3, 358)
    assert batch["text"].shape == (2, 5)
    assert "padding_mask" not in batch


def test_dataset_can_select_only_frozen_semantic_units(tmp_path: Path) -> None:
    row = _row()
    row["raw_text"] = "\u5c0f\u738b\u5b50\u770b\u65e5\u843d"
    spans = list(
        iter_chineseeeg1_character_spans(
            [row],
            record_partitions={"record-1": "train"},
            timeline_audit={"allowed_timeline_methods": ["event_affine"]},
            spec=CharacterSpanSpec(
                span_lengths=(3,),
                neural_delay_ms=0.0,
                annotate_semantic_units=True,
                include_low_confidence=True,
            ),
        )
    )
    index = tmp_path / "semantic-spans.parquet"
    write_character_span_parquet(index, spans)

    semantic = ChineseEEG1SpanDataset(
        index,
        partition="train",
        span_char_count=3,
        semantic_only=True,
        eeg_reader=lambda _file, start, stop: np.ones((2, stop - start)),
    )
    unfiltered = ChineseEEG1SpanDataset(
        index,
        partition="train",
        span_char_count=3,
        eeg_reader=lambda _file, start, stop: np.ones((2, stop - start)),
    )
    assert 0 < len(semantic) < len(unfiltered)
    assert all(semantic[index]["is_semantic_unit"] for index in range(len(semantic)))
