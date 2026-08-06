"""Evaluate a static D-SigLIP checkpoint on the full unique test vocabulary."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from data.chineseeeg1_span_dataset import ChineseEEG1SpanDataset
from evaluation.balanced_text_retrieval import evaluate_balanced_text_retrieval
from evaluation.retrieval_metrics import (
    expected_random_retrieval_metrics,
    metrics_from_ranks,
)
from evaluation.seen_text_retrieval import diagnostic_rank_from_full_rank
from evaluation.seen_text_shortcuts import evaluate_seen_text_shortcuts
from features.consolidated_text_cache import ConsolidatedStaticSpanTextTargetProvider
from models.text_retrieval_factory import build_eeg_text_retrieval_model
from preprocessing.eeg import ChannelRobustScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--span-index", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--max-queries", type=int)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_book_ids(checkpoint: dict[str, object]) -> tuple[str, ...] | None:
    values = checkpoint.get("book_ids")
    if values is None:
        selection = checkpoint["config"].get("selection", {})
        values = selection.get("books")
    if values is None:
        return None
    return tuple(str(value) for value in values)


def _relative(observed: object, random: object) -> dict[str, float]:
    return {
        "recall_at_1_fold": observed.recall_at_1 / random.recall_at_1,
        "recall_at_5_fold": observed.recall_at_5 / random.recall_at_5,
        "recall_at_10_fold": observed.recall_at_10 / random.recall_at_10,
        "mrr_fold": observed.mean_reciprocal_rank / random.mean_reciprocal_rank,
    }


def main() -> None:
    args = parse_args()
    ranks_path = args.output.with_suffix(".query_ranks.jsonl")
    for path in (args.output, ranks_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite test artifact: {path}")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "eeg_text_static_d_siglip":
        raise ValueError("Checkpoint is not a static D-SigLIP model")
    config = checkpoint["config"]
    span_length = int(checkpoint["span_length"])
    span_index = args.span_index or Path(checkpoint["span_index"])
    feature_dir = args.feature_dir or Path(checkpoint["feature_dir"])
    provider = ConsolidatedStaticSpanTextTargetProvider(feature_dir)
    scaler = ChannelRobustScaler.from_state_dict(checkpoint["eeg_scaler"])
    semantic_only = bool(checkpoint["semantic_only"])
    book_ids = _checkpoint_book_ids(checkpoint)
    test_dataset = ChineseEEG1SpanDataset(
        span_index,
        partition="test",
        span_char_count=span_length,
        eeg_normalization="train_recording_robust_clamp",
        eeg_scaler=scaler,
        semantic_only=semantic_only,
        book_ids=book_ids,
    )
    sample = test_dataset[0]
    first_row = test_dataset.table.slice(0, 1).to_pylist()[0]
    first_text = provider(first_row)
    model = build_eeg_text_retrieval_model(
        config,
        eeg_channels=int(sample["eeg"].shape[0]),
        text_shape=first_text.shape,
        n_subjects=len(test_dataset.subject_index_by_group),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    evaluation_config = config["evaluation"]
    result = evaluate_balanced_text_retrieval(
        model,
        test_dataset,
        text_target_provider=provider,
        device=device,
        query_batch_size=int(evaluation_config["query_batch_size"]),
        candidate_batch_size=int(evaluation_config["candidate_batch_size"]),
        max_queries=args.max_queries,
    )
    random = expected_random_retrieval_metrics(result.candidate_count)
    diagnostics: dict[str, object] = {}
    diagnostic_ranks: dict[int, list[int]] = {}
    for requested in evaluation_config["diagnostic_candidate_pool_sizes"]:
        size = int(requested)
        ranks = [
            diagnostic_rank_from_full_rank(
                full_rank=rank,
                full_candidate_count=result.candidate_count,
                requested_pool_size=size,
                seed=int(config["experiment"]["seed"]),
                query_id=query_id,
            )
            for rank, query_id in zip(result.ranks, result.query_ids, strict=True)
        ]
        diagnostic_ranks[size] = ranks
        metrics = metrics_from_ranks(ranks)
        reference = expected_random_retrieval_metrics(
            min(size, result.candidate_count)
        )
        diagnostics[str(size)] = {
            "actual_candidate_count": min(size, result.candidate_count),
            "metrics": asdict(metrics),
            "random_reference": asdict(reference),
            "relative_to_random": _relative(metrics, reference),
        }

    shortcuts = None
    if args.max_queries is None:
        train_dataset = ChineseEEG1SpanDataset(
            span_index,
            partition="train",
            span_char_count=span_length,
            eeg_normalization="none",
            semantic_only=semantic_only,
            book_ids=book_ids,
        )
        shortcut_result = evaluate_seen_text_shortcuts(
            train_dataset.table.to_pylist(),
            test_dataset.table.to_pylist(),
            semantic_only=semantic_only,
            seed=int(config["experiment"]["seed"]),
        )
        shortcuts = {
            "query_count": shortcut_result.query_count,
            "candidate_count": shortcut_result.candidate_count,
            "metrics": {
                key: asdict(value) for key, value in shortcut_result.metrics.items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ranks_path.open("x", encoding="utf-8", newline="\n") as handle:
        for index, query_id in enumerate(result.query_ids):
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "span_text_id": result.target_ids[index],
                        "subject_group_id": result.subject_ids[index],
                        "rank_full": result.ranks[index],
                        **{
                            f"rank_diagnostic_{size}": ranks[index]
                            for size, ranks in diagnostic_ranks.items()
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    report = {
        "schema_version": "ce1-static-d-siglip-test-v1",
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "checkpoint_validation_macro_recall_at_10": float(
            checkpoint["validation_macro_recall_at_10"]
        ),
        "span_length": span_length,
        "book_ids": book_ids,
        "span_index": span_index.as_posix(),
        "query_count": result.query_count,
        "candidate_count": result.candidate_count,
        "micro": asdict(result.micro),
        "macro_class_balanced": asdict(result.macro),
        "random_reference": asdict(random),
        "relative_to_random_micro": _relative(result.micro, random),
        "diagnostic_candidate_pools": diagnostics,
        "shortcut_baselines": shortcuts,
        "query_ranks": ranks_path.as_posix(),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
