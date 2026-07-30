from pathlib import Path

import numpy as np
import pytest

from features.text_features import (
    TextFeatureInput,
    assemble_text_features,
    load_text_features,
    save_text_features,
)


def _result():
    return assemble_text_features(
        item=TextFeatureInput(content_id="content-1", text="你，好"),
        model_id="fake-bert",
        layer_index=-1,
        sentence_pooling="mean_content_tokens",
        token_ids=np.asarray([101, 1, 2, 3, 102, 0]),
        tokens=["[CLS]", "你", "，", "好", "[SEP]", "[PAD]"],
        offsets=np.asarray([[0, 0], [0, 1], [1, 2], [2, 3], [0, 0], [0, 0]]),
        attention_mask=np.asarray([1, 1, 1, 1, 1, 0]),
        special_tokens_mask=np.asarray([1, 0, 0, 0, 1, 1]),
        hidden_states=np.asarray(
            [
                [100.0, 100.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [200.0, 200.0],
                [999.0, 999.0],
            ]
        ),
        truncated=False,
    )


def test_text_features_preserve_offsets_and_exclude_padding_from_pool() -> None:
    result = _result()
    assert result.tokens == ("[CLS]", "你", "，", "好", "[SEP]")
    assert result.token_indices.tolist() == [0, 1, 2, 3, 4]
    assert result.character_offsets.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert result.characters == ("你", "，", "好")
    assert result.character_is_highlighted.tolist() == [True, False, True]
    assert result.character_token_indices == ((1,), (2,), (3,))
    assert result.sentence_hidden_state == pytest.approx([3.0, 4.0])


def test_official_equivalent_pool_includes_special_tokens_but_not_batch_padding() -> None:
    result = assemble_text_features(
        item=TextFeatureInput(content_id="content-1", text="你"),
        model_id="fake-bert",
        layer_index=-1,
        sentence_pooling="mean_attended_tokens",
        token_ids=np.asarray([101, 1, 102, 0]),
        tokens=["[CLS]", "你", "[SEP]", "[PAD]"],
        offsets=np.asarray([[0, 0], [0, 1], [0, 0], [0, 0]]),
        attention_mask=np.asarray([1, 1, 1, 0]),
        special_tokens_mask=np.asarray([1, 0, 1, 1]),
        hidden_states=np.asarray(
            [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [1000.0, 1000.0]]
        ),
        truncated=False,
    )
    assert result.sentence_hidden_state == pytest.approx([3.0, 3.0])


def test_text_feature_npz_roundtrip_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "text.npz"
    original = _result()
    save_text_features(path, original)
    restored = load_text_features(path)
    assert restored.content_id == original.content_id
    assert restored.tokens == original.tokens
    assert restored.characters == original.characters
    assert np.array_equal(
        restored.character_hidden_states,
        original.character_hidden_states,
    )
