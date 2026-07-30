"""Build deterministic, protocol-bound PL EEG/audio physical-time windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data.pl_speech import (
    PLSpeechWindowSpec,
    build_pl_speech_windows,
    load_pl_manifest_rows,
    load_record_partitions,
    render_pl_window_audit_markdown,
    write_pl_window_jsonl,
)
from data.protocol_splitting import write_json_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/pl_speech_retrieval.yaml",
    )
    parser.add_argument("--manifest", default="metadata/all_trials.parquet")
    parser.add_argument(
        "--split-artifact",
        default="splits/text_unseen_seed42.json",
    )
    parser.add_argument("--window-sec", type=float)
    parser.add_argument("--stride-sec", type=float)
    parser.add_argument("--delay-ms", type=float)
    parser.add_argument("--output")
    parser.add_argument("--audit-json")
    parser.add_argument("--audit-markdown")
    return parser.parse_args()


def _delay_label(delay_ms: float) -> str:
    sign = "m" if delay_ms < 0 else ""
    magnitude = abs(delay_ms)
    if magnitude.is_integer():
        return f"{sign}{int(magnitude):03d}ms"
    return f"{sign}{magnitude:.3f}".replace(".", "p") + "ms"


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    window_config = config["window"]
    window_sec = (
        args.window_sec
        if args.window_sec is not None
        else float(window_config["length_sec"])
    )
    stride_sec = (
        args.stride_sec
        if args.stride_sec is not None
        else float(window_config["stride_sec"])
    )
    delay_ms = (
        args.delay_ms
        if args.delay_ms is not None
        else float(window_config["active_eeg_delay_ms"])
    )
    label = _delay_label(delay_ms)
    window_label = f"{window_sec:g}s".replace(".", "p")
    output = Path(
        args.output
        or f"metadata/pl_speech_windows_seed42_{window_label}_delay_{label}.jsonl"
    )
    audit_json = Path(
        args.audit_json
        or f"reports/pl_speech_windows_seed42_{window_label}_delay_{label}.json"
    )
    audit_markdown = Path(
        args.audit_markdown
        or f"reports/pl_speech_windows_seed42_{window_label}_delay_{label}.md"
    )
    rows = load_pl_manifest_rows(args.manifest)
    assignments, _ = load_record_partitions(args.split_artifact)
    windows, audit = build_pl_speech_windows(
        rows,
        record_partitions=assignments,
        spec=PLSpeechWindowSpec(
            window_sec=window_sec,
            stride_sec=stride_sec,
            delay_ms=delay_ms,
        ),
        manifest_path=args.manifest,
        split_artifact_path=args.split_artifact,
    )
    audit["window_jsonl_path"] = output.as_posix()
    audit["window_jsonl_sha256"] = write_pl_window_jsonl(output, windows)
    write_json_deterministic(audit_json, audit)
    audit_markdown.parent.mkdir(parents=True, exist_ok=True)
    audit_markdown.write_text(
        render_pl_window_audit_markdown(audit),
        encoding="utf-8",
        newline="\n",
    )
    print(f"windows={len(windows)}")
    print(f"jsonl={output} sha256={audit['window_jsonl_sha256']}")


if __name__ == "__main__":
    main()
