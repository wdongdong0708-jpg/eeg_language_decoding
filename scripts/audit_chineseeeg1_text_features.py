"""Audit the formal ChineseEEG1 text-feature caches produced for stage two."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from features.cache import safe_artifact_filename
from features.text_features import load_text_features


@dataclass(frozen=True, slots=True)
class FeatureJob:
    name: str
    input_jsonl: Path
    output_dir: Path
    representation_source: str
    layer_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("experiments/features/chineseeeg1_text"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path(
            "reports/generated/chineseeeg1_text_feature_extraction_audit.json"
        ),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path(
            "reports/generated/chineseeeg1_text_feature_extraction_audit.md"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_inputs(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            content_id = str(row["content_id"])
            text = str(row["text"])
            if content_id in rows:
                raise ValueError(
                    f"Duplicate content_id={content_id!r} at {path}:{line_number}"
                )
            rows[content_id] = text
    return rows


def _select_sample_ids(
    content_ids: list[str], *, sample_size: int, seed: int
) -> list[str]:
    ordered = sorted(content_ids)
    if sample_size >= len(ordered):
        return ordered
    anchors = {ordered[0], ordered[len(ordered) // 2], ordered[-1]}
    remaining = [content_id for content_id in ordered if content_id not in anchors]
    random_sample = random.Random(seed).sample(
        remaining, sample_size - len(anchors)
    )
    return sorted(anchors | set(random_sample))


def _validate_feature_sample(
    *,
    job: FeatureJob,
    content_id: str,
    expected_text: str,
) -> dict[str, object]:
    path = job.output_dir / safe_artifact_filename(content_id)
    result = load_text_features(path)
    errors: list[str] = []
    if result.content_id != content_id:
        errors.append("content_id_mismatch")
    if result.text != expected_text:
        errors.append("text_mismatch")
    if result.model_id != "bert-base-chinese":
        errors.append("model_id_mismatch")
    if result.representation_source != job.representation_source:
        errors.append("representation_source_mismatch")
    if result.layer_index != job.layer_index:
        errors.append("layer_index_mismatch")
    if result.sentence_pooling != "mean_content_tokens":
        errors.append("sentence_pooling_mismatch")
    if result.truncated:
        errors.append("unexpected_truncation")
    if result.unmapped_character_indices:
        errors.append("unmapped_characters")
    if result.sentence_hidden_state.shape != (768,):
        errors.append("sentence_shape_mismatch")
    if result.token_hidden_states.shape[1:] != (768,):
        errors.append("token_hidden_shape_mismatch")
    if result.character_hidden_states.shape != (len(expected_text), 768):
        errors.append("character_hidden_shape_mismatch")
    arrays = (
        result.sentence_hidden_state,
        result.token_hidden_states,
        result.character_hidden_states,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        errors.append("nonfinite_values")
    expected_character_indices = np.arange(len(expected_text), dtype=np.int64)
    if not np.array_equal(result.character_indices, expected_character_indices):
        errors.append("character_indices_mismatch")
    if tuple(expected_text) != result.characters:
        errors.append("character_text_mismatch")
    return {
        "content_id": content_id,
        "filename": path.name,
        "text_length": len(expected_text),
        "token_count": len(result.tokens),
        "character_count": len(result.characters),
        "errors": errors,
    }


def _audit_job(
    job: FeatureJob,
    *,
    inputs: dict[str, str],
    sample_ids: list[str],
) -> dict[str, object]:
    manifest_path = job.output_dir / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_paths = list(job.output_dir.glob("*.npz"))
    actual_filenames = {path.name for path in feature_paths}
    expected_filenames = {
        safe_artifact_filename(content_id) for content_id in inputs
    }
    zero_byte_files = [path.name for path in feature_paths if path.stat().st_size == 0]
    samples = [
        _validate_feature_sample(
            job=job,
            content_id=content_id,
            expected_text=inputs[content_id],
        )
        for content_id in sample_ids
    ]
    checks = {
        "manifest_complete": manifest.get("complete") is True,
        "manifest_schema": (
            manifest.get("schema_version") == "text-feature-extraction-run-v1"
        ),
        "input_sha256": manifest.get("input_sha256") == _sha256(job.input_jsonl),
        "input_count": manifest.get("input_count") == len(inputs),
        "feature_count": len(feature_paths) == len(inputs),
        "manifest_feature_count": manifest.get("feature_file_count") == len(inputs),
        "filename_coverage": actual_filenames == expected_filenames,
        "no_zero_byte_files": not zero_byte_files,
        "model_id": manifest.get("configuration", {}).get("model_id")
        == "bert-base-chinese",
        "representation_source": manifest.get("configuration", {}).get(
            "representation_source"
        )
        == job.representation_source,
        "layer_index": manifest.get("configuration", {}).get("layer_index")
        == job.layer_index,
        "max_length": manifest.get("configuration", {}).get("max_length") == 64,
        "strict_character_alignment": manifest.get("configuration", {}).get(
            "strict_character_alignment"
        )
        is True,
        "local_files_only": manifest.get("local_files_only") is True,
        "sample_payloads": all(not sample["errors"] for sample in samples),
    }
    return {
        "name": job.name,
        "input_jsonl": job.input_jsonl.as_posix(),
        "output_dir": job.output_dir.as_posix(),
        "input_count": len(inputs),
        "feature_file_count": len(feature_paths),
        "feature_bytes": sum(path.stat().st_size for path in feature_paths),
        "missing_filename_count": len(expected_filenames - actual_filenames),
        "unexpected_filename_count": len(actual_filenames - expected_filenames),
        "zero_byte_file_count": len(zero_byte_files),
        "sample_size": len(samples),
        "checks": checks,
        "samples": samples,
    }


def _cross_representation_checks(
    jobs: list[FeatureJob], sample_ids_by_input: dict[Path, list[str]]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    scopes = {
        "local": [job for job in jobs if job.name.startswith("local_")],
        "sentence": [job for job in jobs if job.name.startswith("sentence_")],
    }
    for scope, scope_jobs in scopes.items():
        sample_ids = sample_ids_by_input[scope_jobs[0].input_jsonl]
        consistent = True
        for content_id in sample_ids:
            results = [
                load_text_features(
                    job.output_dir / safe_artifact_filename(content_id)
                )
                for job in scope_jobs
            ]
            reference = results[0]
            for result in results[1:]:
                consistent &= result.text == reference.text
                consistent &= result.tokens == reference.tokens
                consistent &= np.array_equal(
                    result.token_ids, reference.token_ids
                )
                consistent &= np.array_equal(
                    result.token_offsets, reference.token_offsets
                )
                consistent &= np.array_equal(
                    result.character_offsets, reference.character_offsets
                )
                consistent &= (
                    result.character_token_indices
                    == reference.character_token_indices
                )
        checks[f"{scope}_offset_provenance_consistent"] = bool(consistent)
    return checks


def _render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# ChineseEEG1 formal text-feature extraction audit",
        "",
        f"- Overall status: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- Total feature files: {report['total_feature_files']:,}",
        f"- Total size: {report['total_feature_bytes'] / 1024**3:.2f} GiB",
        f"- Deterministic validation seed: {report['seed']}",
        "",
        "| Job | Features | Size (GiB) | Sampled | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for job in report["jobs"]:
        passed = all(job["checks"].values())
        lines.append(
            f"| `{job['name']}` | {job['feature_file_count']:,} | "
            f"{job['feature_bytes'] / 1024**3:.2f} | {job['sample_size']} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Cross-representation provenance",
            "",
        ]
    )
    for name, passed in report["cross_representation_checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "The audit verifies complete deterministic filename coverage, non-empty "
            "artifacts, extraction manifests and input hashes. It also loads a "
            "seeded sample from every cache and checks finite 768-dimensional "
            "states, exact input text, character indices, tokenizer offsets and "
            "cross-representation provenance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.sample_size < 3:
        raise ValueError("--sample-size must be at least 3")
    local_input = Path(
        "metadata/generated/chineseeeg1_text_inputs_local_all.jsonl"
    )
    sentence_input = Path(
        "metadata/generated/chineseeeg1_text_inputs_sentence.jsonl"
    )
    jobs = [
        FeatureJob(
            "local_hidden_last",
            local_input,
            args.feature_root / "local_hidden_last",
            "hidden_state",
            -1,
        ),
        FeatureJob(
            "local_hidden_m4",
            local_input,
            args.feature_root / "local_hidden_m4",
            "hidden_state",
            -4,
        ),
        FeatureJob(
            "local_static",
            local_input,
            args.feature_root / "local_static",
            "input_token_embedding",
            0,
        ),
        FeatureJob(
            "sentence_hidden_last",
            sentence_input,
            args.feature_root / "sentence_hidden_last",
            "hidden_state",
            -1,
        ),
        FeatureJob(
            "sentence_hidden_m4",
            sentence_input,
            args.feature_root / "sentence_hidden_m4",
            "hidden_state",
            -4,
        ),
        FeatureJob(
            "sentence_static",
            sentence_input,
            args.feature_root / "sentence_static",
            "input_token_embedding",
            0,
        ),
    ]
    inputs_by_path = {
        path: _load_inputs(path) for path in {local_input, sentence_input}
    }
    sample_ids_by_input = {
        path: _select_sample_ids(
            list(inputs), sample_size=args.sample_size, seed=args.seed
        )
        for path, inputs in inputs_by_path.items()
    }
    job_reports = [
        _audit_job(
            job,
            inputs=inputs_by_path[job.input_jsonl],
            sample_ids=sample_ids_by_input[job.input_jsonl],
        )
        for job in jobs
    ]
    cross_checks = _cross_representation_checks(jobs, sample_ids_by_input)
    passed = all(
        all(job_report["checks"].values()) for job_report in job_reports
    ) and all(cross_checks.values())
    report = {
        "schema_version": "chineseeeg1-text-feature-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "seed": args.seed,
        "sample_size_per_job": args.sample_size,
        "total_feature_files": sum(
            job["feature_file_count"] for job in job_reports
        ),
        "total_feature_bytes": sum(job["feature_bytes"] for job in job_reports),
        "cross_representation_checks": cross_checks,
        "jobs": job_reports,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.report_md.write_text(
        _render_markdown(report), encoding="utf-8", newline="\n"
    )
    print(
        f"status={'PASS' if passed else 'FAIL'} "
        f"files={report['total_feature_files']} "
        f"bytes={report['total_feature_bytes']} "
        f"report={args.report_json}",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
