"""Build all configured PL delays and a matched common-support sweep."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from data.pl_delay_sweep import common_delay_support
from data.pl_speech import (
    PLSpeechWindowSpec,
    build_pl_speech_windows,
    load_pl_manifest_rows,
    load_record_partitions,
    write_pl_window_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/pl_speech_retrieval.yaml",
    )
    parser.add_argument("--output-dir", default="experiments/pl_delay_sweep")
    parser.add_argument(
        "--summary",
        default="reports/pl_speech_delay_sweep_support_seed42.json",
    )
    return parser.parse_args()


def _label(delay: float) -> str:
    if delay.is_integer():
        return f"{int(delay):03d}ms"
    return f"{delay:.3f}".replace(".", "p") + "ms"


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    window_config = config["window"]
    delays = [float(value) for value in window_config["eeg_delay_search_ms"]]
    if len(delays) != len(set(delays)):
        raise ValueError("Configured EEG delays must be unique")
    rows = load_pl_manifest_rows(config["data"]["manifest"])
    assignments, _ = load_record_partitions(config["data"]["split_artifact"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_by_delay = {}
    raw_audits = {}
    for delay in delays:
        windows, audit = build_pl_speech_windows(
            rows,
            record_partitions=assignments,
            spec=PLSpeechWindowSpec(
                window_sec=float(window_config["length_sec"]),
                stride_sec=float(window_config["stride_sec"]),
                delay_ms=delay,
            ),
            manifest_path=config["data"]["manifest"],
            split_artifact_path=config["data"]["split_artifact"],
        )
        label = _label(delay)
        raw_path = output_dir / f"windows_delay_{label}.jsonl"
        audit["window_jsonl_path"] = raw_path.as_posix()
        audit["window_jsonl_sha256"] = write_pl_window_jsonl(raw_path, windows)
        windows_by_delay[delay] = windows
        raw_audits[delay] = audit

    common = common_delay_support(windows_by_delay)
    common_artifacts = {}
    for delay, windows in common.items():
        label = _label(delay)
        path = output_dir / f"windows_delay_{label}_common.jsonl"
        digest = write_pl_window_jsonl(path, windows)
        counts = Counter(window.split for window in windows)
        common_artifacts[str(delay)] = {
            "path": path.as_posix(),
            "sha256": digest,
            "window_count": len(windows),
            "partition_counts": {
                partition: counts.get(partition, 0)
                for partition in ("train", "validation", "test")
            },
            "audio_target_count": len(
                {window.audio_target_id for window in windows}
            ),
            "content_group_count": len(
                {window.split_group_id for window in windows}
            ),
        }
    summary = {
        "schema_version": "pl-delay-common-support-v1",
        "config": Path(args.config).as_posix(),
        "delays_ms": delays,
        "selection_partition": window_config["delay_selection_partition"],
        "test_evaluated_during_delay_selection": False,
        "raw_window_counts": {
            str(delay): len(windows_by_delay[delay]) for delay in sorted(delays)
        },
        "common_support_window_count": len(next(iter(common.values()))),
        "common_support_artifacts": common_artifacts,
        "scientific_reason": (
            "Every delay is trained and validated on identical record/audio/window "
            "units; otherwise longer delays would be compared on a shorter, biased "
            "eligibility subset."
        ),
        "raw_exclusion_counts": {
            str(delay): {
                reason: len(record_ids)
                for reason, record_ids in raw_audits[delay][
                    "excluded_by_reason"
                ].items()
            }
            for delay in sorted(delays)
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary["raw_window_counts"], sort_keys=True))
    print(f"common_support={summary['common_support_window_count']}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
