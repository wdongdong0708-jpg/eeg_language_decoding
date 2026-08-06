"""Extract short-text BERT character states into length-specific NPY arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from features.consolidated_text_cache import CONSOLIDATED_TEXT_CACHE_SCHEMA
from features.text_features import (
    TextEmbeddingExtractor,
    TextFeatureConfig,
    TextFeatureInput,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="bert-base-chinese")
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def _load_inputs(path: Path, maximum: int | None) -> list[TextFeatureInput]:
    items: list[TextFeatureInput] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            item = TextFeatureInput(
                content_id=str(row["content_id"]),
                text=str(row["text"]),
            )
            item.validate()
            if len(item.text) not in {2, 3, 4, 5}:
                raise ValueError("Consolidated cache only supports 2-5 character text")
            items.append(item)
            if maximum is not None and len(items) >= maximum:
                break
    if not items:
        raise ValueError("No text inputs were loaded")
    if len({item.content_id for item in items}) != len(items):
        raise ValueError("Input content IDs must be unique")
    return items


def main() -> None:
    args = parse_args()
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("--max-items must be positive")
    items = _load_inputs(args.input_jsonl, args.max_items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TextFeatureConfig(
        model_id=args.model_id,
        layer_index=args.layer_index,
        sentence_pooling="mean_content_tokens",
        max_length=8,
        batch_size=args.batch_size,
        output_dtype="float32",
    )
    extractor = TextEmbeddingExtractor.from_pretrained(
        config,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    index_rows: list[dict[str, object]] = []
    started = time.monotonic()
    for length in (2, 3, 4, 5):
        selected = [item for item in items if len(item.text) == length]
        if not selected:
            continue
        temporary_path = args.output_dir / f"character_hidden_k{length}.partial.npy"
        final_path = args.output_dir / f"character_hidden_k{length}.npy"
        array = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(selected), length, 768),
        )
        for row_index, result in enumerate(extractor.iter_extract(selected)):
            states = np.asarray(result.character_hidden_states, dtype=np.float32)
            if states.shape != (length, 768):
                raise ValueError(
                    f"Unexpected character state shape for {result.content_id}: "
                    f"{states.shape}"
                )
            array[row_index] = states
            index_rows.append(
                {
                    "content_id": result.content_id,
                    "text": result.text,
                    "span_char_count": length,
                    "row_index": row_index,
                }
            )
            completed = row_index + 1
            if completed % 10_000 == 0 or completed == len(selected):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"k={length} completed={completed}/{len(selected)} "
                    f"global_rate={len(index_rows) / elapsed:.1f}_items_per_sec",
                    flush=True,
                )
        array.flush()
        del array
        os.replace(temporary_path, final_path)
    index_table = pa.Table.from_pylist(index_rows)
    pq.write_table(
        index_table,
        args.output_dir / "feature_index.parquet",
        compression="zstd",
    )
    manifest = {
        "schema_version": CONSOLIDATED_TEXT_CACHE_SCHEMA,
        "complete": True,
        "input_jsonl": args.input_jsonl.as_posix(),
        "input_sha256": hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest(),
        "input_count": len(items),
        "feature_count": len(index_rows),
        "configuration": asdict(config),
        "resolved_model_path": extractor.resolved_model_path,
        "local_files_only": args.local_files_only,
        "storage_dtype": "float32",
        "arrays": {
            str(length): f"character_hidden_k{length}.npy"
            for length in (2, 3, 4, 5)
            if (args.output_dir / f"character_hidden_k{length}.npy").is_file()
        },
    }
    (args.output_dir / "extraction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"complete features={len(index_rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
