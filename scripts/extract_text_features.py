"""Extract sentence, token and character hidden states from JSONL text inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
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
    parser.add_argument(
        "--representation-source",
        choices=["hidden_state", "input_token_embedding"],
        default="hidden_state",
        help=(
            "input_token_embedding bypasses contextual layers and is the static "
            "character/token baseline; use --layer-index 0."
        ),
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed non-empty NPZ files and continue atomically.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print cumulative progress after this many newly written items.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help=(
            "Strict offline mode: resolve a cached snapshot directory and "
            "forbid Hugging Face Hub HTTP access."
        ),
    )
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
        representation_source=args.representation_source,
    )
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = _load_inputs(args.input_jsonl)
    total_count = len(items)
    existing_count = 0
    pending: list[TextFeatureInput] = []
    for item in items:
        target = args.output_dir / safe_artifact_filename(item.content_id)
        if args.resume and target.is_file() and target.stat().st_size > 0:
            existing_count += 1
        else:
            pending.append(item)
    print(
        f"input_count={total_count} existing_count={existing_count} "
        f"pending_count={len(pending)}",
        flush=True,
    )
    extractor = TextEmbeddingExtractor.from_pretrained(
        config,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    started = time.monotonic()
    written_count = 0
    for result in extractor.iter_extract(pending):
        filename = safe_artifact_filename(result.content_id)
        save_text_features(args.output_dir / filename, result)
        written_count += 1
        if written_count % args.progress_every == 0 or written_count == len(pending):
            elapsed = max(time.monotonic() - started, 1e-9)
            completed = existing_count + written_count
            print(
                f"completed={completed}/{total_count} newly_written={written_count} "
                f"rate={written_count / elapsed:.2f}_items_per_sec",
                flush=True,
            )
    feature_files = [
        path
        for path in args.output_dir.glob("*.npz")
        if path.is_file() and path.stat().st_size > 0
    ]
    if len(feature_files) != total_count:
        raise RuntimeError(
            f"Feature file count mismatch: {len(feature_files)} != {total_count}"
        )
    input_sha256 = hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "text-feature-extraction-run-v1",
        "complete": True,
        "input_jsonl": args.input_jsonl.as_posix(),
        "input_sha256": input_sha256,
        "input_count": total_count,
        "feature_file_count": len(feature_files),
        "configuration": asdict(config),
        "resolved_model_path": extractor.resolved_model_path,
        "local_files_only": args.local_files_only,
        "resume_existing_count": existing_count,
        "newly_written_count": written_count,
    }
    (args.output_dir / "extraction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"complete output_dir={args.output_dir} files={len(feature_files)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
