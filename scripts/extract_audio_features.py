"""Extract frame-level wav2vec hidden states for pre-split audio blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from features.audio_features import (
    AudioFeatureConfig,
    AudioFeatureInput,
    Wav2VecFrameExtractor,
    save_audio_frame_features,
)
from features.cache import safe_artifact_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
        help=(
            "Rows require block_id, content_id, split, audio_path, start_sec and stop_sec."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-id",
        default="airesearch/wav2vec2-large-xlsr-53-th",
    )
    parser.add_argument("--layer-index", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _load_inputs(path: Path) -> list[AudioFeatureInput]:
    fields = {
        "block_id",
        "content_id",
        "split",
        "audio_path",
        "start_sec",
        "stop_sec",
    }
    items: list[AudioFeatureInput] = []
    seen_block_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = fields - row.keys()
            if missing:
                raise ValueError(
                    f"Missing {sorted(missing)} at {path}:{line_number}"
                )
            item = AudioFeatureInput(
                block_id=str(row["block_id"]),
                content_id=str(row["content_id"]),
                split=str(row["split"]),
                audio_path=str(row["audio_path"]),
                start_sec=float(row["start_sec"]),
                stop_sec=float(row["stop_sec"]),
            )
            item.validate()
            if item.block_id in seen_block_ids:
                raise ValueError(
                    f"Duplicate block_id={item.block_id!r} at "
                    f"{path}:{line_number}"
                )
            seen_block_ids.add(item.block_id)
            items.append(item)
    return items


def main() -> None:
    args = parse_args()
    config = AudioFeatureConfig(
        model_id=args.model_id,
        layer_index=args.layer_index,
    )
    extractor = Wav2VecFrameExtractor.from_pretrained(
        config,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in _load_inputs(args.input_jsonl):
        result = extractor.extract(item)
        filename = safe_artifact_filename(item.block_id)
        save_audio_frame_features(args.output_dir / filename, result)


if __name__ == "__main__":
    main()
