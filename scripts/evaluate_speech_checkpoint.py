"""One-time partition evaluation of a selected PL EEG–speech checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from data.pl_speech_dataset import PLSpeechDataset
from evaluation.speech_retrieval import evaluate_speech_retrieval
from models.eeg_encoder import DilatedSimpleConv
from models.losses import ClipContrastiveLoss
from models.retrieval_model import EEGSpeechRetrievalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--partition", default="test")
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
    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = PLSpeechDataset(
        training_report["window_index"],
        partition=args.partition,
        feature_dir=config["audio_feature_dir"],
        eeg_normalization=config["eeg_normalization"],
        cache_speech_targets=True,
    )
    sample = dataset[0]
    encoder_config = config["model"]
    encoder = DilatedSimpleConv(
        input_channels=int(sample["eeg"].shape[0]),
        output_channels=int(sample["speech"].shape[0]),
        hidden_channels=int(encoder_config["hidden_channels"]),
        depth=int(encoder_config["depth"]),
        kernel_size=int(encoder_config["kernel_size"]),
        growth=float(encoder_config["growth"]),
        dilation_growth=int(encoder_config["dilation_growth"]),
        dilation_period=encoder_config.get("dilation_period"),
        dropout=float(encoder_config["dropout"]),
        dropout_input=float(encoder_config["dropout_input"]),
        batch_norm=bool(encoder_config["batch_norm"]),
        residual=bool(encoder_config["residual"]),
        activation_on_last=bool(encoder_config["activation_on_last"]),
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
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
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
    report = {
        "schema_version": "pl-speech-checkpoint-evaluation-v1",
        "partition": args.partition,
        "training_report": Path(args.training_report).as_posix(),
        "training_report_sha256": _sha256(args.training_report),
        "checkpoint": Path(args.checkpoint).as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "window_index": training_report["window_index"],
        "window_index_sha256": training_report["window_index_sha256"],
        "active_eeg_delay_ms": training_report["active_eeg_delay_ms"],
        "query_count": len(ranks["global"]),
        "candidate_count": candidate_count,
        "position_local_pool_size": pool_size,
        "tie_policy": "pessimistic",
        "metrics": {
            name: asdict(value) for name, value in metrics.items()
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
