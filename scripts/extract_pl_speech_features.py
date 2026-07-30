"""Extract unpooled multi-layer wav2vec targets for unique PL audio windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from data.pl_speech import PLSpeechWindow, load_pl_window_jsonl
from features.audio_features import (
    DEFAULT_AUDIO_LAYERS,
    AudioFeatureConfig,
    AudioFeatureInput,
    Wav2VecFrameExtractor,
    load_audio_sequence_features,
    save_audio_sequence_features,
)
from features.cache import safe_artifact_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-jsonl",
        default="metadata/pl_speech_windows_seed42_3s_delay_000ms.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/features/pl_speech_wav2vec_3s",
    )
    parser.add_argument(
        "--model-id",
        default="airesearch/wav2vec2-large-xlsr-53-th",
    )
    parser.add_argument(
        "--layers",
        default=",".join(str(value) for value in DEFAULT_AUDIO_LAYERS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit")
    return parser.parse_args()


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise ValueError("--layers cannot be empty")
    return layers


def _unique_targets(
    windows: list[PLSpeechWindow],
) -> list[PLSpeechWindow]:
    targets: dict[str, PLSpeechWindow] = {}
    signatures: dict[str, tuple[object, ...]] = {}
    for window in windows:
        signature = (
            window.audio_file,
            window.audio_start_sample,
            window.audio_stop_sample,
            window.audio_source_sample_rate_hz,
            window.split,
            window.split_group_id,
            window.eeg_sample_count,
        )
        previous = signatures.setdefault(window.audio_target_id, signature)
        if previous != signature:
            raise ValueError(
                f"Inconsistent source for audio_target_id={window.audio_target_id}"
            )
        targets.setdefault(window.audio_target_id, window)
    return [targets[target_id] for target_id in sorted(targets)]


def main() -> None:
    args = parse_args()
    windows = load_pl_window_jsonl(args.window_jsonl)
    targets = _unique_targets(windows)
    if args.max_targets is not None:
        if args.max_targets < 1:
            raise ValueError("--max-targets must be positive")
        targets = targets[: args.max_targets]
    config = AudioFeatureConfig(
        model_id=args.model_id,
        layer_indices=_parse_layers(args.layers),
    )
    extractor = Wav2VecFrameExtractor.from_pretrained(
        config,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = Counter()
    records: list[dict[str, object]] = []
    for target in targets:
        output_path = output_dir / safe_artifact_filename(target.audio_target_id)
        if output_path.exists() and not args.overwrite:
            metadata, _ = load_audio_sequence_features(output_path)
            if metadata["audio_target_id"] != target.audio_target_id:
                raise ValueError(f"Cache target mismatch: {output_path}")
            status["existing"] += 1
        else:
            result = extractor.extract(
                AudioFeatureInput(
                    block_id=target.audio_target_id,
                    content_id=target.split_group_id,
                    split=target.split,
                    audio_path=target.audio_file,
                    start_sec=target.audio_start_sec,
                    stop_sec=target.audio_stop_sec,
                )
            )
            save_audio_sequence_features(
                output_path,
                audio_target_id=target.audio_target_id,
                result=result,
                target_time_steps=target.eeg_sample_count,
            )
            status["extracted"] += 1
        records.append(
            {
                "audio_target_id": target.audio_target_id,
                "split": target.split,
                "split_group_id": target.split_group_id,
                "feature_path": output_path.as_posix(),
            }
        )
    index_path = output_dir / "feature_index.jsonl"
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    index_path.write_text(serialized, encoding="utf-8", newline="\n")
    audit_path = Path(args.audit or output_dir / "feature_audit.json")
    audit = {
        "schema_version": "pl-audio-feature-audit-v1",
        "window_jsonl": Path(args.window_jsonl).as_posix(),
        "window_jsonl_sha256": hashlib.sha256(
            Path(args.window_jsonl).read_bytes()
        ).hexdigest(),
        "model_id": config.model_id,
        "resolved_model_path": extractor.resolved_model_path,
        "layer_indices": list(config.layer_indices),
        "layer_reduction": "arithmetic_mean",
        "temporal_pooling": False,
        "target_time_steps": sorted(
            {target.eeg_sample_count for target in targets}
        ),
        "target_count": len(records),
        "target_counts_by_partition": dict(
            sorted(Counter(record["split"] for record in records).items())
        ),
        "feature_index": index_path.as_posix(),
        "feature_index_sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
        "status_counts": dict(sorted(status.items())),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(dict(sorted(status.items())), sort_keys=True))
    print(f"feature_index={index_path}")
    print(f"feature_audit={audit_path}")


if __name__ == "__main__":
    main()
