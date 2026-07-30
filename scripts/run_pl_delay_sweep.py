"""Train configured PL delays on common support and select by validation loss."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/pl_speech_retrieval.yaml",
    )
    parser.add_argument(
        "--support-summary",
        default="reports/pl_speech_delay_sweep_support_seed42.json",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/pl_delay_sweep/models",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--report-json",
        default="reports/pl_speech_delay_sweep_seed42.json",
    )
    parser.add_argument(
        "--report-markdown",
        default="reports/pl_speech_delay_sweep_seed42.md",
    )
    parser.add_argument("--reuse-matching-reports", action="store_true")
    return parser.parse_args()


def _label(delay: float) -> str:
    if delay.is_integer():
        return f"{int(delay):03d}ms"
    return f"{delay:.3f}".replace(".", "p") + "ms"


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    support = json.loads(
        Path(args.support_summary).read_text(encoding="utf-8")
    )
    delays = [float(value) for value in support["delays_ms"]]
    output_root = Path(args.output_root)
    results = {}
    for delay in delays:
        support_item = support["common_support_artifacts"][str(delay)]
        window_index = support_item["path"]
        output_dir = output_root / f"delay_{_label(delay)}"
        report_path = output_dir / "report.json"
        reusable = False
        if args.reuse_matching_reports and report_path.is_file():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            reusable = (
                previous.get("window_index_sha256") == support_item["sha256"]
                and previous.get("epochs_run") == args.epochs
                and previous.get("active_eeg_delay_ms") == delay
            )
        if not reusable:
            command = [
                sys.executable,
                "scripts/train_speech_retrieval.py",
                "--config",
                args.config,
                "--window-index",
                window_index,
                "--feature-dir",
                config["audio_feature_dir"],
                "--output-dir",
                str(output_dir),
                "--epochs",
                str(args.epochs),
            ]
            subprocess.run(command, check=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        results[str(delay)] = {
            "active_eeg_delay_ms": delay,
            "window_index": window_index,
            "window_index_sha256": report["window_index_sha256"],
            "epochs_run": report["epochs_run"],
            "best_epoch": report["best_epoch"],
            "best_validation_loss": report["best_validation_loss"],
            "validation_retrieval": report["validation_retrieval"],
            "determinism": report["determinism"],
            "checkpoint": (output_dir / "checkpoint.pt").as_posix(),
        }

    train_schedules = {
        item["determinism"]["train_epoch0_batch_schedule_sha256"]
        for item in results.values()
    }
    validation_schedules = {
        item["determinism"]["validation_epoch0_batch_schedule_sha256"]
        for item in results.values()
    }
    if len(train_schedules) != 1 or len(validation_schedules) != 1:
        raise RuntimeError(
            "Common-support delay runs did not use identical batch schedules"
        )

    selected_delay = min(
        delays,
        key=lambda delay: (
            results[str(delay)]["best_validation_loss"],
            delay,
        ),
    )
    summary = {
        "schema_version": "pl-delay-sweep-result-v1",
        "config": Path(args.config).as_posix(),
        "support_summary": Path(args.support_summary).as_posix(),
        "common_support_window_count": support["common_support_window_count"],
        "selection_partition": "validation",
        "selection_metric": "minimum best_validation_loss",
        "selected_delay_ms": selected_delay,
        "test_model_evaluation_status": "not_run",
        "identical_batch_schedule_across_delays": True,
        "results": results,
        "limitations": [
            "Single seed and short pilot schedule; delay selection is provisional.",
            "Position and character-count shortcuts remain strong in validated PL data.",
            "No model result from the test partition was inspected during selection.",
        ],
    }
    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = [
        "# PL EEG-delay sweep",
        "",
        f"- Common-support windows: {support['common_support_window_count']:,}",
        "- Selection partition: validation",
        "- Test model evaluation: not run",
        f"- Provisional selected delay: {selected_delay:g} ms",
        "",
        "| delay | best epoch | validation loss | global R@1 | "
        "position-local R@1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for delay in delays:
        item = results[str(delay)]
        markdown.append(
            f"| {delay:g} ms | {item['best_epoch']} | "
            f"{item['best_validation_loss']:.6f} | "
            f"{item['validation_retrieval']['global']['recall_at_1']:.4f} | "
            f"{item['validation_retrieval']['position_local']['recall_at_1']:.4f} |"
        )
    markdown.extend(
        [
            "",
            "This is a single-seed engineering sweep, not a final scientific result. "
            "The selected delay must be confirmed with a predeclared schedule before "
            "one-time test evaluation.",
            "",
        ]
    )
    report_markdown = Path(args.report_markdown)
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )
    print(f"selected_delay_ms={selected_delay:g}")
    print(f"report={report_json}")


if __name__ == "__main__":
    main()
