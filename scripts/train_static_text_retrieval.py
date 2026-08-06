"""Train pooled ChineseEEG1 EEG-text retrieval with D-SigLIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.chineseeeg1_span_dataset import (
    ChineseEEG1SpanDataset,
    OfficialBrainVisionSegmentReader,
    SpanTextTargetProvider,
    collate_fixed_character_spans,
)
from evaluation.balanced_text_retrieval import evaluate_balanced_text_retrieval
from features.consolidated_text_cache import (
    CONSOLIDATED_TEXT_CACHE_SCHEMA,
    ConsolidatedStaticSpanTextTargetProvider,
)
from models.retrieval_model import EEGTextRetrievalModel
from models.text_retrieval_factory import build_eeg_text_retrieval_model
from preprocessing.eeg import fit_channel_robust_scaler
from training.samplers import AllOccurrenceBatchSampler
from training.text_negatives import (
    NegativePolicyConfig,
    TextCandidate,
    build_pairwise_negative_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "configs/experiment/"
            "chineseeeg1_static_dsiglip_endctx1s_seed42.yaml"
        ),
    )
    parser.add_argument("--span-index", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--span-length", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--max-validation-queries", type=int)
    parser.add_argument("--output-dir", type=Path)
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


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text_provider(feature_dir: Path) -> object:
    manifest_path = feature_dir / "extraction_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") == CONSOLIDATED_TEXT_CACHE_SCHEMA:
            return ConsolidatedStaticSpanTextTargetProvider(feature_dir)
    return SpanTextTargetProvider(feature_dir=feature_dir, mode="local_mean")



def _negative_config(config: dict[str, object]) -> NegativePolicyConfig:
    section = config["loss"]
    return NegativePolicyConfig(
        false_negative_strategy=str(section["false_negative_strategy"]),
        overlap_threshold=float(section["overlap_threshold"]),
        lexical_overlap_threshold=float(section["lexical_overlap_threshold"]),
        semantic_similarity_threshold=float(section["semantic_similarity_threshold"]),
        adjacent_position_distance=int(section["adjacent_position_distance"]),
    )


def _candidates(
    metadata: list[dict[str, object]], text: torch.Tensor
) -> list[TextCandidate]:
    semantic = text.detach().float().cpu().numpy()
    return [
        TextCandidate(
            candidate_id=f"{row['span_event_id']}::{row['record_id']}",
            span_event_id=str(row["span_event_id"]),
            span_text_id=str(row["span_text_id"]),
            global_text_id=str(row["global_text_id"]),
            span_text=str(row["span_text"]),
            span_char_count=int(row["span_char_count"]),
            span_start_clock=int(row["span_start_clock"]),
            span_end_clock=int(row["span_end_clock"]),
            book_id=str(row["book_id"]),
            stimulus_position=int(row["stimulus_position"]),
            semantic_embedding=semantic[index],
        )
        for index, row in enumerate(metadata)
    ]


def _run_epoch(
    model: EEGTextRetrievalModel,
    loader: DataLoader,
    *,
    device: torch.device,
    negative_config: NegativePolicyConfig,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    gradient_clip_norm: float | None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    example_count = 0
    batch_count = 0
    relationship_counts: Counter[str] = Counter()
    duplicate_pair_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            eeg = batch["eeg"].to(device, non_blocking=True)
            text = batch["text"].to(device, non_blocking=True)
            subject_indices = batch["subject_index"].to(device, non_blocking=True)
            policy = build_pairwise_negative_policy(
                _candidates(batch["metadata"], batch["text"]),
                config=negative_config,
            )
            for row in policy.relationship_types:
                relationship_counts.update(row)
            positive_weights = torch.from_numpy(policy.positive_weights).to(device)
            candidate_mask = torch.from_numpy(policy.candidate_mask).to(device)
            duplicate_pair_count += int(
                np.count_nonzero(policy.positive_weights) - len(batch["metadata"])
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(
                eeg,
                text,
                subject_indices=subject_indices,
                positive_weights=positive_weights,
                candidate_mask=candidate_mask,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                optimizer.step()
            count = int(eeg.shape[0])
            total_loss += float(loss.detach()) * count
            example_count += count
            batch_count += 1
    if not example_count:
        raise ValueError("DataLoader produced no examples")
    return {
        "loss": total_loss / example_count,
        "example_count": example_count,
        "batch_count": batch_count,
        "duplicate_off_diagonal_pairs_removed_by_d_siglip": duplicate_pair_count,
        "relationship_counts": dict(sorted(relationship_counts.items())),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["loss"]["name"] != "d_siglip":
        raise ValueError("This entry point requires loss.name=d_siglip")
    if config["loss"]["false_negative_strategy"] != "mask":
        raise ValueError("D-SigLIP requires binary positives and mask false negatives")
    training_config = config["training"]
    if str(training_config["optimizer"]).lower() != "adamw":
        raise ValueError("Static D-SigLIP protocol requires AdamW")
    if str(training_config["scheduler"]).lower() != "cosine":
        raise ValueError("Static D-SigLIP protocol requires cosine scheduling")
    if training_config["checkpoint_metric"] != "validation_macro_recall_at_10":
        raise ValueError("Checkpoint metric must be balanced validation Top-10")

    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    device = torch.device(args.device)
    index_path = args.span_index or Path(config["data"]["span_index"])
    feature_dir = args.feature_dir or Path(config["data"]["text_feature_dir"])
    provider = _text_provider(feature_dir)
    semantic_only = bool(config.get("selection", {}).get("semantic_only", False))
    reader = OfficialBrainVisionSegmentReader()

    raw_train_dataset = ChineseEEG1SpanDataset(
        index_path,
        partition="train",
        span_char_count=args.span_length,
        eeg_normalization="none",
        text_target_provider=provider,
        eeg_reader=reader,
        semantic_only=semantic_only,
    )
    preprocessing_config = config["preprocessing"]
    normalization = str(preprocessing_config["eeg_normalization"])
    if normalization != "train_recording_robust_clamp":
        raise ValueError("This protocol requires train_recording_robust_clamp")
    fit_rows = raw_train_dataset.table.select(
        ["split", "eeg_file", "eeg_start_sample", "eeg_stop_sample"]
    ).to_pylist()
    eeg_scaler = fit_channel_robust_scaler(
        fit_rows,
        eeg_reader=reader,
        clamp=float(preprocessing_config["clamp"]),
    )
    dataset_kwargs = {
        "span_char_count": args.span_length,
        "eeg_normalization": normalization,
        "eeg_scaler": eeg_scaler,
        "text_target_provider": provider,
        "eeg_reader": reader,
        "semantic_only": semantic_only,
    }
    train_dataset = ChineseEEG1SpanDataset(
        index_path, partition="train", **dataset_kwargs
    )
    validation_dataset = ChineseEEG1SpanDataset(
        index_path, partition="validation", **dataset_kwargs
    )

    batch_size = int(training_config["batch_size"])
    train_sampler = AllOccurrenceBatchSampler(
        len(train_dataset),
        batch_size=batch_size,
        seed=seed,
        drop_last=bool(training_config["drop_last"]),
    )
    validation_sampler = AllOccurrenceBatchSampler(
        len(validation_dataset),
        batch_size=batch_size,
        seed=seed + 1,
        drop_last=bool(training_config["validation_drop_last"]),
    )
    loader_kwargs = {
        "collate_fn": collate_fixed_character_spans,
        "num_workers": int(training_config["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, batch_sampler=validation_sampler, **loader_kwargs
    )

    sample = train_dataset[0]
    model = build_eeg_text_retrieval_model(
        config,
        eeg_channels=int(sample["eeg"].shape[0]),
        text_shape=tuple(sample["text"].shape),
        n_subjects=len(train_dataset.subject_index_by_group),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = args.epochs or int(training_config["epochs"])
    if epochs > 50:
        raise ValueError("The configured maximum is 50 epochs")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(training_config.get("minimum_learning_rate", 0.0)),
    )
    gradient_clip_value = training_config.get("gradient_clip_norm")
    gradient_clip_norm = (
        None if gradient_clip_value is None else float(gradient_clip_value)
    )
    max_train_batches = args.max_train_batches
    max_validation_batches = args.max_validation_batches
    max_validation_queries = args.max_validation_queries
    if args.smoke_test:
        epochs = 1
        max_train_batches = max_train_batches or 2
        max_validation_batches = max_validation_batches or 1
        max_validation_queries = max_validation_queries or 256

    output_dir = args.output_dir or Path(config["output"]["directory_template"].format(
        span_length=args.span_length
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_validation_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    patience = int(training_config["early_stopping_patience"])
    negative_config = _negative_config(config)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            negative_config=negative_config,
            optimizer=optimizer,
            max_batches=max_train_batches,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_loss_metrics = _run_epoch(
            model,
            validation_loader,
            device=device,
            negative_config=negative_config,
            optimizer=None,
            max_batches=max_validation_batches,
            gradient_clip_norm=None,
        )
        balanced = evaluate_balanced_text_retrieval(
            model,
            validation_dataset,
            text_target_provider=provider,
            device=device,
            query_batch_size=int(config["evaluation"]["query_batch_size"]),
            candidate_batch_size=int(config["evaluation"]["candidate_batch_size"]),
            max_queries=max_validation_queries,
        )
        score = float(balanced.macro.recall_at_10)
        validation_loss = float(validation_loss_metrics["loss"])
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation_loss": validation_loss_metrics,
            "validation_balanced_retrieval": {
                "candidate_count": balanced.candidate_count,
                "query_count": balanced.query_count,
                "class_count": balanced.class_count,
                "micro": asdict(balanced.micro),
                "macro": asdict(balanced.macro),
            },
            "d_siglip": {
                "logit_scale": float(model.objective.logit_scale.exp().detach()),
                "bias": float(model.objective.bias.detach()),
            },
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        improved = score > best_score or (
            score == best_score and validation_loss < best_validation_loss
        )
        if improved:
            best_score = score
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": "ce1-static-d-siglip-checkpoint-v1",
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "eeg_scaler": eeg_scaler.state_dict(),
                    "eeg_scaler_audit": eeg_scaler.audit(),
                    "config": config,
                    "config_path": config_path.as_posix(),
                    "config_sha256": _sha256(config_path),
                    "span_index": index_path.as_posix(),
                    "span_index_sha256": _sha256(index_path),
                    "feature_dir": feature_dir.as_posix(),
                    "span_length": args.span_length,
                    "text_target_mode": "local_mean_static",
                    "model_kind": "eeg_text_static_d_siglip",
                    "seed": seed,
                    "semantic_only": semantic_only,
                                "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_macro_recall_at_10": score,
                    "validation_balanced_retrieval": record[
                        "validation_balanced_retrieval"
                    ],
                },
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= patience:
            break

    summary = {
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "span_length": args.span_length,
        "train_examples_per_epoch": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_validation_macro_recall_at_10": best_score,
        "best_validation_loss_tiebreak": best_validation_loss,
        "epochs_completed": len(history),
        "checkpoint_metric": "validation_macro_recall_at_10",
        "eeg_scaler": eeg_scaler.audit(),
        "smoke_test": args.smoke_test,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
