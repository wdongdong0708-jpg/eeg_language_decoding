"""Build Stage-2 ChineseEEG1 fixed 4/6/8-character EEG span indices."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from data.chineseeeg1_spans import (
    CharacterSpanSpec,
    audit_span_index,
    iter_chineseeeg1_character_spans,
    load_protocol_record_partitions,
    render_span_audit_markdown,
    write_character_span_parquet,
)

SOURCE_COLUMNS = [
    "dataset_version",
    "paradigm",
    "subject_id",
    "session_id",
    "book_id",
    "chapter_id",
    "sentence_id",
    "global_text_id",
    "raw_text",
    "eeg_file",
    "eeg_start_sample",
    "eeg_end_sample",
    "eeg_sampling_rate",
    "quality_flag",
    "split_group_id",
    "record_id",
    "run_id",
    "block_id",
    "content_id",
    "stimulus_position",
    "text_alignment_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/chineseeeg1_text_retrieval.yaml",
    )
    parser.add_argument("--manifest", default="metadata/all_trials.parquet")
    parser.add_argument("--split-artifact", default="splits/text_unseen_seed42.json")
    parser.add_argument(
        "--timeline-audit",
        default="reports/chineseeeg1_character_timeline_audit.json",
    )
    parser.add_argument(
        "--output",
        default="metadata/generated/chineseeeg1_character_spans_seed42.parquet",
    )
    parser.add_argument(
        "--audit-json",
        default="reports/chineseeeg1_character_spans_seed42.json",
    )
    parser.add_argument(
        "--audit-markdown",
        default="reports/chineseeeg1_character_spans_seed42.md",
    )
    parser.add_argument("--span-lengths", nargs="+", type=int)
    parser.add_argument("--stride-characters", type=int)
    parser.add_argument("--span-start-clocks", nargs="+", type=int)
    parser.add_argument("--neural-delay-ms", type=float)
    parser.add_argument("--left-context-ms", type=float)
    parser.add_argument("--right-context-ms", type=float)
    parser.add_argument("--include-low-confidence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    span_config = config["span_index"]
    spec = CharacterSpanSpec(
        span_lengths=tuple(args.span_lengths or span_config["span_lengths"]),
        stride_characters=(
            args.stride_characters
            if args.stride_characters is not None
            else int(span_config["stride_characters"])
        ),
        allowed_clock_starts=(
            tuple(args.span_start_clocks)
            if args.span_start_clocks is not None
            else (
                tuple(int(value) for value in span_config["span_start_clocks"])
                if span_config.get("span_start_clocks") is not None
                else None
            )
        ),
        timeline_method=str(span_config["timeline_method"]),
        neural_delay_ms=(
            args.neural_delay_ms
            if args.neural_delay_ms is not None
            else float(span_config["neural_delay_ms"])
        ),
        left_context_ms=(
            args.left_context_ms
            if args.left_context_ms is not None
            else float(span_config["left_context_ms"])
        ),
        right_context_ms=(
            args.right_context_ms
            if args.right_context_ms is not None
            else float(span_config["right_context_ms"])
        ),
        target_character_duration_ms=float(
            span_config["target_character_duration_ms"]
        ),
        include_low_confidence=(
            args.include_low_confidence
            or bool(span_config.get("include_low_confidence", False))
        ),
    )
    spec.validate()
    manifest_table = pq.read_table(args.manifest, columns=SOURCE_COLUMNS)
    rows = manifest_table.to_pylist()
    assignments, _ = load_protocol_record_partitions(args.split_artifact)
    timeline_audit = json.loads(Path(args.timeline_audit).read_text(encoding="utf-8"))
    counters: Counter[str] = Counter()
    spans = iter_chineseeeg1_character_spans(
        rows,
        record_partitions=assignments,
        timeline_audit=timeline_audit,
        spec=spec,
        counters=counters,
    )
    count, digest = write_character_span_parquet(args.output, spans)
    audit = audit_span_index(
        args.output,
        counters=counters,
        spec=spec,
        timeline_audit_path=args.timeline_audit,
        split_artifact_path=args.split_artifact,
    )
    Path(args.audit_json).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    Path(args.audit_markdown).write_text(
        render_span_audit_markdown(audit),
        encoding="utf-8",
        newline="\n",
    )
    print(f"spans={count}")
    print(f"parquet={args.output} sha256={digest}")


if __name__ == "__main__":
    main()
