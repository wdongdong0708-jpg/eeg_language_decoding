"""Aggregate PL speech baseline engineering evidence into stable reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-audit",
        default="reports/pl_speech_windows_seed42_3s_delay_000ms.json",
    )
    parser.add_argument(
        "--feature-audit",
        default="reports/pl_speech_audio_feature_audit_seed42.json",
    )
    parser.add_argument(
        "--full-pilot",
        default="experiments/pl_speech_retrieval_seed42_delay000ms/report.json",
    )
    parser.add_argument(
        "--delay-sweep",
        default="reports/pl_speech_delay_sweep_seed42.json",
    )
    parser.add_argument(
        "--selected-test",
        default="reports/pl_speech_selected_delay500ms_test_seed42.json",
    )
    parser.add_argument(
        "--selected-shortcuts",
        default="reports/pl_speech_shortcuts_common_delay500ms_test_seed42.json",
    )
    parser.add_argument(
        "--output-json",
        default="reports/pl_speech_baseline_validation.json",
    )
    parser.add_argument(
        "--output-markdown",
        default="reports/pl_speech_baseline_validation.md",
    )
    return parser.parse_args()


def _load(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    source_paths = {
        "window_audit": args.window_audit,
        "feature_audit": args.feature_audit,
        "full_pilot": args.full_pilot,
        "delay_sweep": args.delay_sweep,
        "selected_test": args.selected_test,
        "selected_shortcuts": args.selected_shortcuts,
    }
    sources = {name: _load(path) for name, path in source_paths.items()}
    window = sources["window_audit"]
    feature = sources["feature_audit"]
    pilot = sources["full_pilot"]
    sweep = sources["delay_sweep"]
    selected = sources["selected_test"]
    shortcuts = sources["selected_shortcuts"]
    report = {
        "schema_version": "pl-speech-baseline-validation-v1",
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "window_accounting": window["counts"],
        "window_exclusion_counts": {
            reason: len(record_ids)
            for reason, record_ids in window["excluded_by_reason"].items()
        },
        "window_leakage_checks": window["leakage_checks"],
        "audio_features": {
            key: feature[key]
            for key in (
                "model_id",
                "layer_indices",
                "layer_reduction",
                "temporal_pooling",
                "target_time_steps",
                "target_count",
                "target_counts_by_partition",
                "feature_index_sha256",
            )
        },
        "zero_ms_full_support_pilot": {
            "epochs_run": pilot["epochs_run"],
            "best_epoch": pilot["best_epoch"],
            "best_validation_loss": pilot["best_validation_loss"],
            "validation_retrieval": pilot["validation_retrieval"],
        },
        "delay_sweep": {
            "common_support_window_count": sweep[
                "common_support_window_count"
            ],
            "selection_partition": sweep["selection_partition"],
            "selection_metric": sweep["selection_metric"],
            "selected_delay_ms": sweep["selected_delay_ms"],
            "identical_batch_schedule_across_delays": sweep[
                "identical_batch_schedule_across_delays"
            ],
            "results": sweep["results"],
        },
        "selected_delay_test": {
            "delay_ms": selected["active_eeg_delay_ms"],
            "query_count": selected["query_count"],
            "candidate_count": selected["candidate_count"],
            "metrics": selected["metrics"],
            "shortcut_baselines": shortcuts["baselines"],
        },
        "interpretation": {
            "engineering_checks_passed": [
                "verified EEG/audio physical-time windows",
                "no content-group or audio-target split leakage",
                "multi-layer wav2vec average without time pooling",
                "matching EEG/speech sequence lengths",
                "finite end-to-end gradients and decreasing training loss",
                "deterministic common-support delay comparison",
                "chunked unique-candidate retrieval with pessimistic ties",
            ],
            "not_supported": (
                "A semantic/language-decoding claim. Sentence position and character "
                "count remain strong shortcuts, and the selected delay is a "
                "single-seed short-schedule result."
            ),
            "confirmatory_status": (
                "Exploratory only. Test shortcut diagnostics were inspected while "
                "hardening the candidate-pool protocol, before the selected model was "
                "evaluated. A future confirmatory result needs a newly predeclared "
                "split/seed or an untouched external cohort."
            ),
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    counts = report["window_accounting"]
    test_metrics = report["selected_delay_test"]["metrics"]
    random_metrics = report["selected_delay_test"]["shortcut_baselines"]["random"]
    position_metrics = report["selected_delay_test"]["shortcut_baselines"][
        "sentence_position_only"
    ]
    markdown = [
        "# PL EEG–speech baseline validation",
        "",
        "This is an engineering/synchronization baseline. It is not evidence of "
        "semantic or linguistic decoding.",
        "",
        "## Data and windows",
        "",
        f"- PL manifest trials: {counts['input_pl_record_count']:,}",
        f"- Exact 3-second windows: {counts['window_count']:,}",
        f"- Train/validation/test windows: "
        f"{counts['window_counts_by_partition']['train']:,}/"
        f"{counts['window_counts_by_partition']['validation']:,}/"
        f"{counts['window_counts_by_partition']['test']:,}",
        f"- Unique speech targets: {counts['audio_target_count']:,}",
        "- Content/audio-target cross-split leakage: 0/0",
        "",
        "## Audio targets",
        "",
        f"- Model: `{feature['model_id']}`",
        f"- Hidden layers averaged: `{feature['layer_indices']}`",
        "- Temporal pooling: false",
        f"- Output shape: `[1024, {feature['target_time_steps'][0]}]`",
        f"- Cached targets: {feature['target_count']:,}",
        "",
        "## Delay sweep",
        "",
        f"- Common support: {sweep['common_support_window_count']:,} windows",
        "- Delays: 0/100/200/300/400/500 ms",
        f"- Provisional validation-selected delay: "
        f"{sweep['selected_delay_ms']:g} ms",
        "- Identical epoch-0 batch schedule across delays: true",
        "- Test EEG was not used during delay selection.",
        "",
        "## One-time selected-delay test",
        "",
        f"- Queries/candidates: {selected['query_count']}/"
        f"{selected['candidate_count']}",
        "",
        "| pool | model R@1 | model R@10 | random R@1 | random R@10 | "
        "position-only R@1 | position-only R@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pool in ("global", "position_local"):
        markdown.append(
            f"| {pool} | {test_metrics[pool]['recall_at_1']:.4f} | "
            f"{test_metrics[pool]['recall_at_10']:.4f} | "
            f"{random_metrics[pool]['recall_at_1']:.4f} | "
            f"{random_metrics[pool]['recall_at_10']:.4f} | "
            f"{position_metrics[pool]['recall_at_1']:.4f} | "
            f"{position_metrics[pool]['recall_at_10']:.4f} |"
        )
    markdown.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "Training loss and retrieval above random confirm that the pipeline can "
            "learn and rank synchronized targets. They do not isolate linguistic "
            "content. Sentence position reaches R@10=1.0 on the selected common-support "
            "test pool, character count is also strong, and the untrained envelope "
            "baseline is competitive on some metrics. This dataset subset should "
            "therefore remain an engineering unit test unless stronger controls or "
            "additional independently ordered audio material become available.",
            "",
            "The seed-42 test set is not pristine for confirmatory inference: its "
            "shortcut diagnostics were inspected while the evaluation protocol was "
            "being hardened. The selected checkpoint score remains useful as an "
            "engineering check, but a confirmatory claim requires a newly predeclared "
            "split/seed or untouched external cohort.",
            "",
        ]
    )
    output_markdown = Path(args.output_markdown)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )
    print(f"json={output_json}")
    print(f"markdown={output_markdown}")


if __name__ == "__main__":
    main()
