"""Extract sentence, token and character hidden states from JSONL text inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from features.cache import safe_artifact_filename
from features.text_features import (
    TextEmbeddingExtractor,
    TextFeatureConfig,
    TextFeatureInput,
    save_text_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
        help="Rows require content_id and text.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="bert-base-chinese")
    parser.add_argument(
        "--sentence-pooling",
        choices=["mean_content_tokens", "mean_attended_tokens", "cls"],
        default="mean_content_tokens",
    )
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _load_inputs(path: Path) -> list[TextFeatureInput]:
    items: list[TextFeatureInput] = []
    seen_content_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                item = TextFeatureInput(
                    content_id=str(row["content_id"]),
                    text=str(row["text"]),
                )
            except KeyError as error:
                raise ValueError(
                    f"Missing {error.args[0]} at {path}:{line_number}"
                ) from error
            item.validate()
            if item.content_id in seen_content_ids:
                raise ValueError(
                    f"Duplicate content_id={item.content_id!r} at "
                    f"{path}:{line_number}"
                )
            seen_content_ids.add(item.content_id)
            items.append(item)
    return items


def main() -> None:
    args = parse_args()
    config = TextFeatureConfig(
        model_id=args.model_id,
        layer_index=args.layer_index,
        sentence_pooling=args.sentence_pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    extractor = TextEmbeddingExtractor.from_pretrained(
        config,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = _load_inputs(args.input_jsonl)
    for result in extractor.extract(items):
        filename = safe_artifact_filename(result.content_id)
        save_text_features(args.output_dir / filename, result)


if __name__ == "__main__":
    main()
