import numpy as np

from features.text_features import (
    TextFeatureInput,
    assemble_text_features,
    pool_text_span,
    select_text_span_by_offsets,
)


def test_span_selection_uses_offsets_and_explicit_clock_indices() -> None:
    result = assemble_text_features(
        item=TextFeatureInput(content_id="global-1", text="甲，乙丙"),
        model_id="fake",
        layer_index=-1,
        sentence_pooling="mean_content_tokens",
        token_ids=np.asarray([101, 1, 2, 3, 102]),
        tokens=["[CLS]", "甲，", "乙", "丙", "[SEP]"],
        offsets=np.asarray([[0, 0], [0, 2], [2, 3], [3, 4], [0, 0]]),
        attention_mask=np.ones(5),
        special_tokens_mask=np.asarray([1, 0, 0, 0, 1]),
        hidden_states=np.asarray(
            [[9.0, 9.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [8.0, 8.0]]
        ),
        truncated=False,
    )
    selection = select_text_span_by_offsets(
        result,
        raw_start=0,
        raw_stop=4,
        included_character_indices=(0, 2, 3),
        expected_character_count=3,
    )
    assert selection.characters == ("甲", "乙", "丙")
    assert selection.source_token_indices == ((1,), (2,), (3,))
    assert pool_text_span(result, selection, pooling="mean").tolist() == [3.0, 4.0]
    assert pool_text_span(result, selection, pooling="cls").tolist() == [9.0, 9.0]
