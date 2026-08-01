"""One-time BrainMagick-style full-candidate test evaluation.

The paper-compatible primary metric pools every test EEG query and ranks it
against every unique test speech segment. Per-subject metrics are additionally
reported without changing that global candidate vocabulary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from data.pl_speech_dataset import PLSpeechDataset
from data.pl_speech_scaling import RecordingRobustScaler, SpeechStandardScaler
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    expected_random_retrieval_metrics,
    metrics_by_group,
)
from evaluation.speech_retrieval import evaluate_speech_retrieval
from models.eeg_encoder import build_eeg_encoder
from models.losses import ClipContrastiveLoss
from models.retrieval_model import EEGSpeechRetrievalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-protocol", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_sha256(directory: str | Path, pattern: str) -> tuple[str, int]:
    rows: list[str] = []
    for path in sorted(Path(directory).glob(pattern), key=lambda item: item.name):
        rows.append(f"{path.name}|{_sha256(path)}\n")
    digest = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    return digest, len(rows)


def _verify_frozen_inputs(protocol: dict[str, object]) -> None:
    for section in ("frozen_inputs", "evaluation_code"):
        for item in protocol[section]:
            path = Path(item["path"])
            observed = _sha256(path)
            if observed != item["sha256"]:
                raise ValueError(
                    f"Frozen {section} artifact changed: {path} "
                    f"({observed} != {item['sha256']})"
                )
    for item in protocol.get("frozen_trees", []):
        observed, count = _tree_sha256(item["path"], item["pattern"])
        if observed != item["sha256"] or count != item["file_count"]:
            raise ValueError(f"Frozen file tree changed: {item['path']}")


def _strip_runtime_overrides(config: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(config)
    normalized.pop("seed", None)
    normalized.pop("output_dir", None)
    return normalized


def _load_and_verify_training_config(
    training_report: dict[str, object],
    protocol: dict[str, object],
) -> dict[str, object]:
    frozen_config_path = Path(protocol["config"]["path"])
    if _sha256(frozen_config_path) != protocol["config"]["sha256"]:
        raise ValueError("Frozen configuration hash no longer matches")
    frozen_config = yaml.safe_load(frozen_config_path.read_text(encoding="utf-8"))
    actual_config = training_report["config"]
    if _strip_runtime_overrides(actual_config) != _strip_runtime_overrides(
        frozen_config
    ):
        raise ValueError(
            "Training report does not match the frozen protocol after allowing "
            "only seed/output_dir runtime overrides"
        )
    seed = int(training_report["seed"])
    if seed not in protocol["training_seeds"]:
        raise ValueError(f"Training seed {seed} is not in the frozen seed set")
    return actual_config


def _write_query_ranks(
    path: Path,
    dataset: PLSpeechDataset,
    ranks: list[int],
) -> None:
    if len(dataset.windows) != len(ranks):
        raise ValueError("Query ranks do not align with test windows")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for window, rank in zip(dataset.windows, ranks, strict=True):
            row = {
                "window_id": window.window_id,
                "audio_target_id": window.audio_target_id,
                "subject_group_id": window.subject_group_id,
                "rank": rank,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    ranks_output = output.with_suffix(".query_ranks.jsonl")
    for artifact in (output, ranks_output):
        if artifact.exists():
            raise FileExistsError(
                f"Refusing to overwrite one-time test artifact: {artifact}"
            )

    protocol_path = Path(args.frozen_protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen":
        raise ValueError("Evaluation requires a frozen protocol")
    _verify_frozen_inputs(protocol)

    training_report_path = Path(args.training_report)
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    config = _load_and_verify_training_config(training_report, protocol)
    window_index = Path(training_report["window_index"])
    if _sha256(window_index) != training_report["window_index_sha256"]:
        raise ValueError("Training window index changed after training")

    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["window_index_sha256"] != training_report["window_index_sha256"]:
        raise ValueError("Checkpoint and training report use different window indices")

    preprocessing_state = checkpoint.get("preprocessing_state", {})
    subject_index_by_group = checkpoint.get("subject_index_by_group")
    if subject_index_by_group is None:
        subject_index_by_group = training_report.get("subjects", {}).get(
            "index_by_group"
        )
    if config["model"].get("subject_layer") and subject_index_by_group is None:
        raise ValueError("Subject-layer checkpoint is missing its subject mapping")
    eeg_scaler_state = preprocessing_state.get("eeg")
    speech_scaler_state = preprocessing_state.get("speech")
    eeg_scaler = (
        RecordingRobustScaler.from_state_dict(eeg_scaler_state)
        if eeg_scaler_state is not None
        else None
    )
    speech_scaler = (
        SpeechStandardScaler.from_state_dict(speech_scaler_state)
        if speech_scaler_state is not None
        else None
    )
    alignment_offset_ms = float(
        training_report.get("alignment", {}).get(
            "offset_ms", config["window"].get("alignment_offset_ms", 0.0)
        )
    )
    dataset = PLSpeechDataset(
        window_index,
        partition="test",
        feature_dir=training_report.get(
            "audio_feature_dir", config["audio_feature_dir"]
        ),
        eeg_normalization=config["eeg_normalization"],
        eeg_scaler=eeg_scaler,
        speech_scaler=speech_scaler,
        expected_audio_model_id=config["audio_target"]["model_id"],
        alignment_offset_ms=alignment_offset_ms,
        cache_speech_targets=True,
        subject_index_by_group=subject_index_by_group,
    )
    if alignment_offset_ms and {window.eeg_delay_ms for window in dataset.windows} != {
        0.0
    }:
        raise ValueError("Internal alignment crop requires zero-delay source windows")

    sample = dataset[0]
    encoder = build_eeg_encoder(
        config["model"],
        input_channels=int(sample["eeg"].shape[0]),
        output_channels=int(sample["speech"].shape[0]),
        n_subjects=len(dataset.subject_index_by_group),
    )
    loss_config = config["loss"]
    model = EEGSpeechRetrievalModel(
        encoder,
        ClipContrastiveLoss(
            norm_kind=loss_config["norm_kind"],
            learn_temperature=bool(loss_config["learn_temperature"]),
            temperature=float(loss_config["temperature"]),
            symmetric=bool(loss_config["symmetric"]),
            center=bool(loss_config["center"]),
            pool=bool(loss_config["pool"]),
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics, ranks, candidate_count = evaluate_speech_retrieval(
        model,
        dataset,
        device=device,
        query_batch_size=int(config["evaluation"]["query_batch_size"]),
        candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
        norm_kind=loss_config["norm_kind"],
        position_pool_size=None,
    )
    global_ranks = ranks["global"]
    subject_ids = [window.subject_group_id for window in dataset.windows]
    per_subject = metrics_by_group(global_ranks, subject_ids)
    subject_macro = aggregate_retrieval_metrics(list(per_subject.values()), ddof=1)
    query_counts = {
        subject_id: subject_ids.count(subject_id) for subject_id in per_subject
    }
    random_baseline = expected_random_retrieval_metrics(candidate_count)

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_query_ranks(ranks_output, dataset, global_ranks)
    report = {
        "schema_version": "brainmagick-full-test-evaluation-v1",
        "protocol_id": protocol["protocol_id"],
        "frozen_protocol": protocol_path.as_posix(),
        "frozen_protocol_sha256": _sha256(protocol_path),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "test",
        "seed": int(training_report["seed"]),
        "training_report": training_report_path.as_posix(),
        "training_report_sha256": _sha256(training_report_path),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "window_index": window_index.as_posix(),
        "window_index_sha256": _sha256(window_index),
        "query_ranks": ranks_output.as_posix(),
        "query_ranks_sha256": _sha256(ranks_output),
        "selection": {
            "criterion": "minimum_validation_loss",
            "best_epoch_zero_based": training_report["best_epoch"],
            "best_validation_loss": training_report["best_validation_loss"],
            "test_used_for_selection": False,
        },
        "protocol": {
            "query_unit": "every_test_eeg_window",
            "candidate_vocabulary": "all_unique_test_audio_target_id",
            "candidate_scope": "global_and_identical_for_every_query_and_subject",
            "primary_aggregation": "micro_average_over_all_test_queries",
            "secondary_aggregation": "per_subject_then_macro_average",
            "tie_policy": "pessimistic",
            "score": "BrainMagick_CLIP_sequence_inner_product",
        },
        "query_count": len(global_ranks),
        "candidate_count": candidate_count,
        "subject_count": len(per_subject),
        "brainmagick_table_compatible": {
            "micro": asdict(metrics["global"]),
        },
        "per_subject": {
            subject_id: {
                "query_count": query_counts[subject_id],
                "metrics": asdict(subject_metrics),
            }
            for subject_id, subject_metrics in per_subject.items()
        },
        "subject_macro": asdict(subject_macro),
        "random_baseline": {
            "method": "analytical_uniform_random_ranking",
            "candidate_count": candidate_count,
            "metrics": asdict(random_baseline),
        },
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["brainmagick_table_compatible"], sort_keys=True))
    print(json.dumps(report["subject_macro"], sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
