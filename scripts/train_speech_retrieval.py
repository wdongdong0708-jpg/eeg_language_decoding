"""Train and validate the ChineseEEG2 PL EEG–speech sequence baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.pl_speech import load_pl_window_jsonl
from data.pl_speech_dataset import BrainVisionSegmentReader, PLSpeechDataset
from data.pl_speech_scaling import (
    fit_recording_robust_scaler,
    fit_speech_standard_scaler,
)
from evaluation.speech_retrieval import evaluate_speech_retrieval
from models.eeg_encoder import build_eeg_encoder
from models.losses import ClipContrastiveLoss
from models.retrieval_model import EEGSpeechRetrievalModel
from training.samplers import UniqueTargetBatchSampler
from training.trainer import evaluate_loss, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/pl_speech_retrieval.yaml",
    )
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--window-index")
    parser.add_argument("--feature-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _batch_schedule_sha256(
    dataset: PLSpeechDataset,
    sampler: UniqueTargetBatchSampler,
) -> str:
    digest = hashlib.sha256()
    sampler.set_epoch(0)
    for batch in sampler:
        for index in batch:
            window = dataset.windows[index]
            payload = (
                f"{window.record_id}\0{window.audio_target_id}\0"
                f"{window.window_offset_sec}\0{window.subject_group_id}\n"
            )
            digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    _seed_everything(seed)
    device = torch.device(
        args.device
        or config.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    window_jsonl = args.window_index or config["window_index"]
    feature_dir = args.feature_dir or config["audio_feature_dir"]
    all_windows = load_pl_window_jsonl(window_jsonl)
    delays = sorted({window.eeg_delay_ms for window in all_windows})
    if len(delays) != 1:
        raise ValueError(f"One training run requires exactly one EEG delay: {delays}")
    window_source: object = all_windows
    if args.smoke_test:
        index_path = Path(feature_dir) / "feature_index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Smoke test requires a partial feature index: {index_path}"
            )
        available_targets = {
            json.loads(line)["audio_target_id"]
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        window_source = [
            window
            for window in all_windows
            if window.audio_target_id in available_targets
        ]
    eeg_scaler = None
    if config["eeg_normalization"] == "train_recording_robust_clamp":
        eeg_scaler = fit_recording_robust_scaler(
            window_source,
            eeg_reader=BrainVisionSegmentReader(),
            clamp=float(config.get("eeg_clamp", 20.0)),
        )
    speech_scaler = None
    speech_target_scaling = config.get("speech_target_scaling", "none")
    if speech_target_scaling == "train_global_standard":
        speech_scaler = fit_speech_standard_scaler(
            window_source,
            feature_dir=feature_dir,
            expected_model_id=config["audio_target"]["model_id"],
        )
    elif speech_target_scaling != "none":
        raise ValueError(
            f"Unknown speech target scaling: {speech_target_scaling}"
        )
    train_dataset = PLSpeechDataset(
        window_source,
        partition="train",
        feature_dir=feature_dir,
        eeg_normalization=config["eeg_normalization"],
        eeg_scaler=eeg_scaler,
        speech_scaler=speech_scaler,
        expected_audio_model_id=config["audio_target"]["model_id"],
        cache_speech_targets=bool(config["training"]["cache_speech_targets"]),
    )
    validation_dataset = PLSpeechDataset(
        window_source,
        partition="validation",
        feature_dir=feature_dir,
        eeg_normalization=config["eeg_normalization"],
        eeg_scaler=eeg_scaler,
        speech_scaler=speech_scaler,
        expected_audio_model_id=config["audio_target"]["model_id"],
        cache_speech_targets=bool(config["training"]["cache_speech_targets"]),
    )
    batch_size = int(config["training"]["batch_size"])
    train_sampler = UniqueTargetBatchSampler(
        train_dataset.audio_target_ids,
        batch_size=batch_size,
        seed=seed,
        drop_last=bool(config["training"].get("drop_last", True)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    validation_batch_size = min(
        int(config["training"].get("validation_batch_size", batch_size)),
        len(set(validation_dataset.audio_target_ids)),
    )
    validation_sampler = UniqueTargetBatchSampler(
        validation_dataset.audio_target_ids,
        batch_size=validation_batch_size,
        seed=seed + 1,
        drop_last=bool(config["training"].get("validation_drop_last", True)),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        num_workers=0,
    )
    train_batch_schedule_sha256 = _batch_schedule_sha256(
        train_dataset,
        train_sampler,
    )
    validation_batch_schedule_sha256 = _batch_schedule_sha256(
        validation_dataset,
        validation_sampler,
    )
    sample = train_dataset[0]
    encoder_config = config["model"]
    encoder = build_eeg_encoder(
        encoder_config,
        input_channels=int(sample["eeg"].shape[0]),
        output_channels=int(sample["speech"].shape[0]),
    )
    loss_config = config["loss"]
    objective = ClipContrastiveLoss(
        norm_kind=loss_config["norm_kind"],
        learn_temperature=bool(loss_config["learn_temperature"]),
        temperature=float(loss_config["temperature"]),
        symmetric=bool(loss_config["symmetric"]),
        center=bool(loss_config["center"]),
        pool=bool(loss_config["pool"]),
    )
    model = EEGSpeechRetrievalModel(encoder, objective).to(device)
    optimizer_name = str(config["training"].get("optimizer", "adamw")).lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = args.epochs or int(config["training"]["epochs"])
    max_batches = 2 if args.smoke_test else None
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_epoch = -1
    best_model_state: dict[str, torch.Tensor] | None = None
    gradient_clip_value = config["training"].get("gradient_clip_norm")
    gradient_clip_norm = (
        None if gradient_clip_value is None else float(gradient_clip_value)
    )
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            max_batches=max_batches,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_result = evaluate_loss(
            model,
            validation_loader,
            device=device,
            max_batches=max_batches,
        )
        entry = {
            "epoch": epoch,
            "train": train_result,
            "validation": validation_result,
        }
        history.append(entry)
        if float(validation_result["loss"]) < best_validation_loss:
            best_validation_loss = float(validation_result["loss"])
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        print(json.dumps(entry, sort_keys=True))

    if best_model_state is None:
        raise RuntimeError("No best model state was selected")
    model.load_state_dict(best_model_state)
    position_pool_size = min(
        int(config["evaluation"]["position_local_pool_size"]),
        len(set(validation_dataset.audio_target_ids)),
    )
    retrieval_metrics, ranks, candidate_count = evaluate_speech_retrieval(
        model,
        validation_dataset,
        device=device,
        query_batch_size=int(config["evaluation"]["query_batch_size"]),
        candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
        norm_kind=loss_config["norm_kind"],
        max_queries=(
            int(config["evaluation"]["smoke_max_queries"])
            if args.smoke_test
            else None
        ),
        position_pool_size=position_pool_size,
    )
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "window_index": window_jsonl,
        "audio_feature_dir": str(feature_dir),
        "active_eeg_delay_ms": delays[0],
        "window_index_sha256": _file_sha256(window_jsonl),
        "config": config,
        "history": history,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "preprocessing": {
            "eeg": eeg_scaler.audit() if eeg_scaler is not None else {
                "name": config["eeg_normalization"],
            },
            "speech": (
                speech_scaler.audit()
                if speech_scaler is not None
                else {"name": speech_target_scaling}
            ),
        },
        "batching": {
            "training_batch_size": batch_size,
            "training_drop_last": train_sampler.drop_last,
            "validation_batch_size": validation_batch_size,
            "validation_drop_last": validation_sampler.drop_last,
        },
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "train_epoch0_batch_schedule_sha256": train_batch_schedule_sha256,
            "validation_epoch0_batch_schedule_sha256": (
                validation_batch_schedule_sha256
            ),
        },
        "validation_retrieval": {
            "global": asdict(retrieval_metrics["global"]),
            "position_local": asdict(retrieval_metrics["position_local"]),
            "query_count": len(ranks["global"]),
            "candidate_count": candidate_count,
            "position_local_pool_size": position_pool_size,
            "tie_policy": "pessimistic",
        },
        "smoke_test": args.smoke_test,
    }
    report_path = output_dir / (
        "smoke_report.json" if args.smoke_test else "report.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checkpoint_path = output_dir / (
        "smoke_checkpoint.pt" if args.smoke_test else "checkpoint.pt"
    )
    torch.save(
        {
            "schema_version": "pl-speech-retrieval-checkpoint-v2",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "preprocessing_state": {
                "eeg": (
                    eeg_scaler.state_dict() if eeg_scaler is not None else None
                ),
                "speech": (
                    speech_scaler.state_dict()
                    if speech_scaler is not None
                    else None
                ),
            },
            "report_path": report_path.as_posix(),
            "window_index_sha256": report["window_index_sha256"],
        },
        checkpoint_path,
    )
    print(f"report={report_path}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
