"""Aggregate three frozen BrainMagick-style test evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from evaluation.retrieval_metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(data: dict[str, float]) -> RetrievalMetrics:
    return RetrievalMetrics(**data)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    markdown_output = Path(args.markdown_output)
    for artifact in (output, markdown_output):
        if artifact.exists():
            raise FileExistsError(f"Refusing to overwrite aggregate: {artifact}")

    input_paths = [Path(path) for path in args.inputs]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    if any(
        report.get("schema_version") != "brainmagick-full-test-evaluation-v1"
        for report in reports
    ):
        raise ValueError("All inputs must be BrainMagick full-test evaluations")
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != 3:
        raise ValueError("Exactly three distinct seeds are required")

    invariant_keys = (
        "protocol_id",
        "frozen_protocol_sha256",
        "partition",
        "window_index_sha256",
        "candidate_count",
        "query_count",
        "subject_count",
    )
    for key in invariant_keys:
        values = {json.dumps(report[key], sort_keys=True) for report in reports}
        if len(values) != 1:
            raise ValueError(f"Evaluation inputs disagree on {key}")

    subject_ids = set(reports[0]["per_subject"])
    if any(set(report["per_subject"]) != subject_ids for report in reports[1:]):
        raise ValueError("Per-subject evaluation keys differ across seeds")

    micro_records = [
        _metrics(report["brainmagick_table_compatible"]["micro"])
        for report in reports
    ]
    subject_macro_records = [
        _metrics(report["subject_macro"]["mean"]) for report in reports
    ]
    micro_aggregate = aggregate_retrieval_metrics(micro_records, ddof=1)
    subject_macro_aggregate = aggregate_retrieval_metrics(
        subject_macro_records, ddof=1
    )

    per_subject_across_seeds: dict[str, object] = {}
    for subject_id in sorted(subject_ids):
        subject_records = [
            _metrics(report["per_subject"][subject_id]["metrics"])
            for report in reports
        ]
        per_subject_across_seeds[subject_id] = {
            "query_count": reports[0]["per_subject"][subject_id]["query_count"],
            "across_seeds": asdict(
                aggregate_retrieval_metrics(subject_records, ddof=1)
            ),
        }

    summary = {
        "schema_version": "brainmagick-three-seed-summary-v1",
        "protocol_id": reports[0]["protocol_id"],
        "frozen_protocol_sha256": reports[0]["frozen_protocol_sha256"],
        "partition": "test",
        "seeds": sorted(seeds),
        "standard_deviation": {
            "kind": "sample",
            "ddof": 1,
            "rationale": "matches pandas std() used by BrainMagick's paper notebook",
        },
        "candidate_count": reports[0]["candidate_count"],
        "query_count": reports[0]["query_count"],
        "subject_count": reports[0]["subject_count"],
        "inputs": [
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "seed": int(report["seed"]),
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
            for path, report in zip(input_paths, reports, strict=True)
        ],
        "per_seed": [
            {
                "seed": int(report["seed"]),
                "micro": report["brainmagick_table_compatible"]["micro"],
                "subject_macro_mean": report["subject_macro"]["mean"],
            }
            for report in sorted(reports, key=lambda item: int(item["seed"]))
        ],
        "brainmagick_table_compatible_across_seeds": asdict(micro_aggregate),
        "subject_macro_across_seeds": asdict(subject_macro_aggregate),
        "per_subject_across_seeds": per_subject_across_seeds,
        "random_baseline": reports[0]["random_baseline"],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    mean = micro_aggregate.mean
    std = micro_aggregate.std
    macro_mean = subject_macro_aggregate.mean
    macro_std = subject_macro_aggregate.std
    chance = _metrics(reports[0]["random_baseline"]["metrics"])
    lines = [
        "# Frozen BrainMagick-style three-seed test evaluation",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in sorted(seeds))}",
        f"- Test queries: {reports[0]['query_count']}",
        f"- Full test candidates: {reports[0]['candidate_count']}",
        f"- Subjects: {reports[0]['subject_count']}",
        "- Dispersion: sample standard deviation across seeds (ddof=1)",
        "",
        "| Aggregation | R@1 | R@5 | R@10 | Median rank | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            "| BrainMagick-compatible micro | "
            f"{_pct(mean.recall_at_1)} ± {_pct(std.recall_at_1)} | "
            f"{_pct(mean.recall_at_5)} ± {_pct(std.recall_at_5)} | "
            f"{_pct(mean.recall_at_10)} ± {_pct(std.recall_at_10)} | "
            f"{mean.median_rank:.2f} ± {std.median_rank:.2f} | "
            f"{mean.mean_reciprocal_rank:.4f} ± {std.mean_reciprocal_rank:.4f} |"
        ),
        (
            "| Per-subject macro | "
            f"{_pct(macro_mean.recall_at_1)} ± {_pct(macro_std.recall_at_1)} | "
            f"{_pct(macro_mean.recall_at_5)} ± {_pct(macro_std.recall_at_5)} | "
            f"{_pct(macro_mean.recall_at_10)} ± {_pct(macro_std.recall_at_10)} | "
            f"{macro_mean.median_rank:.2f} ± {macro_std.median_rank:.2f} | "
            f"{macro_mean.mean_reciprocal_rank:.4f} ± "
            f"{macro_std.mean_reciprocal_rank:.4f} |"
        ),
        (
            "| Analytical random | "
            f"{_pct(chance.recall_at_1)} | {_pct(chance.recall_at_5)} | "
            f"{_pct(chance.recall_at_10)} | {chance.median_rank:.2f} | "
            f"{chance.mean_reciprocal_rank:.4f} |"
        ),
        "",
        "The primary row follows BrainMagick's published aggregation: all test "
        "queries against the same full unique test-segment vocabulary, followed "
        "by mean/sample-standard-deviation across three model seeds. The "
        "per-subject macro row is an additional diagnostic.",
    ]
    markdown_output.write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(asdict(micro_aggregate), sort_keys=True))
    print(f"report={output}")
    print(f"markdown={markdown_output}")


if __name__ == "__main__":
    main()
