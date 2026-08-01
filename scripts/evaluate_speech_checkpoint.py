"""One-time partition evaluation of a selected PL EEG–speech checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from data.pl_speech_dataset import PLSpeechDataset
from data.pl_speech_scaling import (
    RecordingRobustScaler,
    SpeechStandardScaler,
)
from evaluation.speech_retrieval import evaluate_speech_retrieval
from evaluation.retrieval_metrics import expected_random_retrieval_metrics
from models.eeg_encoder import build_eeg_encoder
from models.losses import ClipContrastiveLoss
from models.retrieval_model import EEGSpeechRetrievalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--window-index")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite one-time evaluation report: {output}"
        )
    training_report = json.loads(
        Path(args.training_report).read_text(encoding="utf-8")
    )
    config = training_report["config"]
    evaluation_window_index = (
        args.window_index or training_report["window_index"]
    )
    alignment_offset_ms = float(
        training_report.get("alignment", {}).get(
            "offset_ms",
            config["window"].get("alignment_offset_ms", 0.0),
        )
    )
    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
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
    dataset = PLSpeechDataset(
        evaluation_window_index,
        partition=args.partition,
        feature_dir=training_report.get(
            "audio_feature_dir",
            config["audio_feature_dir"],
        ),
        eeg_normalization=config["eeg_normalization"],
        eeg_scaler=eeg_scaler,
        speech_scaler=speech_scaler,
        expected_audio_model_id=config["audio_target"]["model_id"],
        alignment_offset_ms=alignment_offset_ms,
        cache_speech_targets=True,
        subject_index_by_group=subject_index_by_group,
    )
    source_delays = {window.eeg_delay_ms for window in dataset.windows}
    if alignment_offset_ms and source_delays != {0.0}:
        raise ValueError(
            "Internal alignment crop requires a zero-delay window index"
        )
    sample = dataset[0]
    encoder_config = config["model"]
    encoder = build_eeg_encoder(
        encoder_config,
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
    pool_size = min(
        int(config["evaluation"]["position_local_pool_size"]),
        len(set(dataset.audio_target_ids)),
    )
    metrics, ranks, candidate_count = evaluate_speech_retrieval(
        model,
        dataset,
        device=device,
        query_batch_size=int(config["evaluation"]["query_batch_size"]),
        candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
        norm_kind=loss_config["norm_kind"],
        position_pool_size=pool_size,
    )
    random_global = expected_random_retrieval_metrics(candidate_count)
    random_position_local = expected_random_retrieval_metrics(
        pool_size
    )
    report = {
        "schema_version": "pl-speech-checkpoint-evaluation-v1",
        "partition": args.partition,
        "training_report": Path(args.training_report).as_posix(),
        "training_report_sha256": _sha256(args.training_report),
        "checkpoint": Path(args.checkpoint).as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "training_window_index": training_report["window_index"],
        "window_index": str(evaluation_window_index),
        "window_index_sha256": _sha256(evaluation_window_index),
        "active_eeg_delay_ms": training_report["active_eeg_delay_ms"],
        "alignment": training_report.get("alignment"),
        "query_count": len(ranks["global"]),
        "candidate_count": candidate_count,
        "position_local_pool_size": pool_size,
        "tie_policy": "pessimistic",
        "metrics": {
            name: asdict(value) for name, value in metrics.items()
        },
        "random_baseline": {
            "method": "analytical_uniform_random_ranking",
            "global": asdict(random_global),
            "position_local": asdict(random_position_local),
            "candidate_count": candidate_count,
            "position_local_pool_size": pool_size,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["metrics"], sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
