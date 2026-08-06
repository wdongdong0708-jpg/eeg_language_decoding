"""Build common-support +1 s/+2 s post-span EEG sensitivity indices."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/preprocessing/chineseeeg1_end_context_sweep.yaml",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Write every valid occurrence per condition instead of common support",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extend_rows(
    source_rows: list[dict[str, object]], post_context_ms: float
) -> tuple[list[dict[str, object]], Counter[str]]:
    output: list[dict[str, object]] = []
    counters: Counter[str] = Counter()
    for source in source_rows:
        if float(source["neural_delay_ms"]) != 0.0:
            raise ValueError("End-context sweep expects the zero-delay source index")
        if float(source["left_context_ms"]) != 0.0:
            raise ValueError("End-context sweep expects no source left context")
        if float(source["right_context_ms"]) != 0.0:
            raise ValueError("End-context sweep expects no source right context")
        if str(source["timeline_method"]) != "fixed_dwell_sensitivity":
            raise ValueError("End-context sweep requires the fixed 0.35 s clock")
        row = dict(source)
        sampling_rate = int(row["eeg_sampling_rate_hz"])
        span_length = int(row["span_char_count"])
        sample_count = round(
            (span_length * 350.0 + post_context_ms) / 1000.0 * sampling_rate
        )
        eeg_stop = int(row["eeg_start_sample"]) + sample_count
        if eeg_stop > int(row["source_row_stop_sample"]):
            counters[f"dropped_cross_row:k{span_length}"] += 1
            continue
        row["eeg_stop_sample"] = eeg_stop
        row["source_eeg_sample_count"] = sample_count
        row["model_eeg_sample_count"] = sample_count
        row["right_context_ms"] = float(post_context_ms)
        row["timeline_rule"] = (
            "ROWS + clock_index * 0.35 s; EEG starts at first span character "
            f"and retains {post_context_ms / 1000.0:g} s after the estimated "
            "final-character display boundary"
        )
        row["resampling_method"] = (
            "linear_interpolation_to_fixed_span_plus_end_context_length"
        )
        output.append(row)
        counters[f"retained:k{span_length}"] += 1
    return output, counters


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = Path(config["source_index"])
    source_table = pq.read_table(source_path).combine_chunks()
    source_rows = source_table.to_pylist()
    contexts = tuple(float(value) for value in config["post_context_ms"])
    if not contexts or any(value <= 0 for value in contexts):
        raise ValueError("post_context_ms must contain positive values")
    if len(set(contexts)) != len(contexts):
        raise ValueError("post_context_ms values must be unique")

    rows_by_context: dict[float, list[dict[str, object]]] = {}
    counters_by_context: dict[float, Counter[str]] = {}
    for context in contexts:
        rows, counters = _extend_rows(source_rows, context)
        rows_by_context[context] = rows
        counters_by_context[context] = counters
    common_keys = set.intersection(
        *(
            {
                (str(row["record_id"]), str(row["span_event_id"]))
                for row in rows_by_context[context]
            }
            for context in contexts
        )
    )
    if not common_keys:
        raise ValueError("End-context conditions have no common support")

    outputs: dict[str, object] = {}
    for context in contexts:
        label = f"{int(context):04d}ms"
        template_key = "raw_output_template" if args.raw_only else "output_template"
        output_path = Path(str(config[template_key]).format(label=label))
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite index: {output_path}")
        selected_rows = (
            rows_by_context[context]
            if args.raw_only
            else [
                row
                for row in rows_by_context[context]
                if (str(row["record_id"]), str(row["span_event_id"])) in common_keys
            ]
        )
        table = pa.Table.from_pylist(selected_rows, schema=source_table.schema)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="zstd")
        lengths = sorted(
            {int(row["model_eeg_sample_count"]) for row in selected_rows}
        )
        outputs[label] = {
            "post_context_ms": context,
            "path": output_path.as_posix(),
            "sha256": _sha256(output_path),
            "raw_retained_count": len(rows_by_context[context]),
            "output_support_count": len(selected_rows),
            "model_eeg_sample_counts": lengths,
            "generation_counts": dict(sorted(counters_by_context[context].items())),
        }

    audit_path = Path(
        config["raw_audit_output"] if args.raw_only else config["audit_output"]
    )
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
    audit = {
        "schema_version": "ce1-static-end-context-sweep-v1",
        "source_index": source_path.as_posix(),
        "source_index_sha256": _sha256(source_path),
        "source_count": source_table.num_rows,
        "timeline_assumption": "fixed presentation clock of 0.35 s per timed character",
        "alignment_anchor": "estimated final-character display boundary",
        "window_start": "estimated first-character display onset",
        "cross_sentence_context": "forbidden_and_dropped",
        "common_support_key": ["record_id", "span_event_id"],
        "common_support_count": len(common_keys),
        "output_support": (
            "condition_specific_all_valid_occurrences"
            if args.raw_only
            else "intersection_across_all_context_conditions"
        ),
        "conditions": outputs,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
