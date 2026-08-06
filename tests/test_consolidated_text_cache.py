import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from features.consolidated_text_cache import (
    CONSOLIDATED_TEXT_CACHE_SCHEMA,
    ConsolidatedSpanTextTargetProvider,
)


def test_consolidated_provider_returns_the_requested_character_sequence(tmp_path) -> None:
    values = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    np.save(tmp_path / "character_hidden_k2.npy", values)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "content_id": "text-a",
                    "text": "\u7532\u4e59",
                    "span_char_count": 2,
                    "row_index": 1,
                }
            ]
        ),
        tmp_path / "feature_index.parquet",
    )
    (tmp_path / "extraction_manifest.json").write_text(
        json.dumps(
            {"schema_version": CONSOLIDATED_TEXT_CACHE_SCHEMA, "complete": True}
        ),
        encoding="utf-8",
    )
    provider = ConsolidatedSpanTextTargetProvider(tmp_path)
    result = provider(
        {
            "span_text_id": "text-a",
            "span_text": "\u7532\u4e59",
            "span_char_count": 2,
        }
    )
    assert np.array_equal(result, values[1])
