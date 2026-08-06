"""Evaluate a static EEG-text checkpoint on per-book test-frequency pools.

This protocol is diagnostic: candidate identities are selected using test-label
occurrence frequencies.  EEG scores are not used during candidate selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from data.chineseeeg1_span_dataset import ChineseEEG1SpanDataset
from evaluation.book_stratified_retrieval import (
    RetrievalProtocol,
    evaluate_book_stratified_protocols,
)
from features.consolidated_text_cache import (
    ConsolidatedStaticSpanTextTargetProvider,
)
from models.text_retrieval_factory import build_eeg_text_retrieval_model
from preprocessing.eeg import ChannelRobustScaler


BOOK_NAMES = {
    "littleprince": "小王子",
    "garnettdream": "狼王梦",
}
POOL_SIZES = {
    "littleprince": (50, 100),
    "garnettdream": (50, 100, 150, 200, 250),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
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


def _minimal_rows(dataset: ChineseEEG1SpanDataset) -> list[dict[str, object]]:
    columns = [
        "record_id",
        "span_event_id",
        "span_start_clock",
        "span_text_id",
        "span_text",
        "span_char_count",
        "book_id",
        "subject_group_id",
        "stimulus_position",
        "is_semantic_unit",
    ]
    return dataset.table.select(columns).to_pylist()


def _build_protocols(
    train_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
) -> tuple[RetrievalProtocol, ...]:
    representative_text: dict[str, str] = {}
    for row in (*train_rows, *test_rows):
        representative_text.setdefault(
            str(row["span_text_id"]), str(row["span_text"])
        )

    protocols: list[RetrievalProtocol] = []
    for book_id, pool_sizes in POOL_SIZES.items():
        book_indices = tuple(
            index
            for index, row in enumerate(test_rows)
            if str(row["book_id"]) == book_id
        )
        if not book_indices:
            continue
        counts = Counter(
            str(test_rows[index]["span_text_id"]) for index in book_indices
        )
        ranked_ids = tuple(
            text_id
            for text_id, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        )
        if len(ranked_ids) < max(pool_sizes):
            raise ValueError(
                f"{book_id} has {len(ranked_ids)} unique test texts, fewer "
                f"than the requested {max(pool_sizes)}"
            )

        for pool_size in pool_sizes:
            candidate_ids = ranked_ids[:pool_size]
            candidate_set = set(candidate_ids)
            query_indices = tuple(
                index
                for index in book_indices
                if str(test_rows[index]["span_text_id"]) in candidate_set
            )
            boundary_frequency = counts[candidate_ids[-1]]
            protocols.append(
                RetrievalProtocol(
                    name=f"{book_id}_test_frequency_top{pool_size}",
                    family="static_test_frequency_semantic_diagnostic",
                    candidate_ids=candidate_ids,
                    query_indices=query_indices,
                    query_universe_indices=book_indices,
                    candidate_selection={
                        "method": "within_book_test_eeg_occurrence_frequency",
                        "book_id": book_id,
                        "book_name": BOOK_NAMES[book_id],
                        "partition": "test",
                        "semantic_only": True,
                        "requested_candidate_count": pool_size,
                        "actual_candidate_count": len(candidate_ids),
                        "tie_breaker": "span_text_id_ascending",
                        "uses_test_labels": True,
                        "uses_model_scores": False,
                        "total_book_test_query_count": len(book_indices),
                        "total_book_test_unique_text_count": len(counts),
                        "selected_query_count": len(query_indices),
                        "selected_query_coverage": len(query_indices)
                        / len(book_indices),
                        "minimum_included_test_frequency": boundary_frequency,
                        "vocabulary_types_at_boundary_frequency": sum(
                            value == boundary_frequency for value in counts.values()
                        ),
                        "candidate_details": [
                            {
                                "rank": rank,
                                "span_text_id": text_id,
                                "span_text": representative_text[text_id],
                                "test_occurrence_count": counts[text_id],
                            }
                            for rank, text_id in enumerate(candidate_ids, 1)
                        ],
                    },
                    query_selection={
                        "method": (
                            "all_same_book_test_eeg_occurrences_with_target_in_pool"
                        ),
                        "selected_query_count": len(query_indices),
                        "total_book_test_query_count": len(book_indices),
                    },
                )
            )
    return tuple(protocols)


class _StaticProtocolAdapter(torch.nn.Module):
    """Expose the sequence protocol evaluator's two required model methods."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def encode_eeg(
        self, eeg: torch.Tensor, subject_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.model.encode_eeg(eeg, subject_indices)

    def get_compressed_text_scores(
        self, estimates: torch.Tensor, candidate_text_states: torch.Tensor
    ) -> torch.Tensor:
        candidates = self.model.encode_text(candidate_text_states)
        return self.model.objective.get_scores(estimates, candidates)


def main() -> None:
    args = parse_args()
    ranks_path = args.output.with_suffix(".query_ranks.jsonl")
    for path in (args.output, ranks_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "eeg_text_static_d_siglip":
        raise ValueError("Checkpoint is not a static D-SigLIP model")
    config = checkpoint["config"]
    span_length = int(checkpoint["span_length"])
    span_index = Path(checkpoint["span_index"])
    feature_dir = Path(checkpoint["feature_dir"])
    semantic_only = bool(checkpoint["semantic_only"])
    book_ids = _checkpoint_book_ids(checkpoint)
    if not semantic_only:
        raise ValueError("This word-pool protocol requires a semantic-only checkpoint")

    provider = ConsolidatedStaticSpanTextTargetProvider(feature_dir)
    scaler = ChannelRobustScaler.from_state_dict(checkpoint["eeg_scaler"])
    train_dataset = ChineseEEG1SpanDataset(
        span_index,
        partition="train",
        span_char_count=span_length,
        eeg_normalization="none",
        semantic_only=True,
        book_ids=book_ids,
    )
    test_dataset = ChineseEEG1SpanDataset(
        span_index,
        partition="test",
        span_char_count=span_length,
        eeg_normalization="train_recording_robust_clamp",
        eeg_scaler=scaler,
        semantic_only=True,
        book_ids=book_ids,
    )
    train_rows = _minimal_rows(train_dataset)
    test_rows = _minimal_rows(test_dataset)
    protocols = _build_protocols(train_rows, test_rows)

    representative_by_id: dict[str, dict[str, object]] = {}
    for row in (*train_rows, *test_rows):
        representative_by_id.setdefault(str(row["span_text_id"]), row)
    sample = test_dataset[0]
    first_row = test_dataset.table.slice(0, 1).to_pylist()[0]
    model = build_eeg_text_retrieval_model(
        config,
        eeg_channels=int(sample["eeg"].shape[0]),
        text_shape=provider(first_row).shape,
        n_subjects=len(test_dataset.subject_index_by_group),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    adapter = _StaticProtocolAdapter(model).to(device)
    evaluation = config["evaluation"]
    results, rank_rows = evaluate_book_stratified_protocols(
        adapter,
        test_dataset,
        train_rows,
        test_rows,
        protocols,
        text_target_provider=provider,
        representative_by_text_id=representative_by_id,
        device=device,
        query_batch_size=int(evaluation["query_batch_size"]),
        candidate_batch_size=int(evaluation["candidate_batch_size"]),
        seed=int(config["experiment"]["seed"]),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ranks_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rank_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "ce1-static-test-frequency-by-book-v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_class": "diagnostic_test_label_frequency_selected_candidates",
        "protocol_warning": (
            "Candidate identities are selected from test-label occurrence "
            "frequencies; these are diagnostic, not blind confirmatory results."
        ),
        "metric_definitions": {
            "micro": "all selected EEG query occurrences weighted equally",
            "macro": "Top-K per represented target text, then texts weighted equally",
            "rank_tie_policy": "pessimistic: candidates tied with the positive rank ahead",
        },
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch_zero_based": int(checkpoint["epoch"]),
        "checkpoint_validation_macro_recall_at_10": float(
            checkpoint["validation_macro_recall_at_10"]
        ),
        "span_length": span_length,
        "span_index": span_index.as_posix(),
        "span_index_sha256": _sha256(span_index),
        "semantic_only": semantic_only,
        "book_ids": book_ids,
        "pool_sizes_by_book": {
            book: list(sizes)
            for book, sizes in POOL_SIZES.items()
            if book_ids is None or book in book_ids
        },
        "query_ranks": ranks_path.as_posix(),
        "protocols": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    concise = {
        name: {
            "candidates": values["candidate_count"],
            "queries": values["query_count"],
            "micro": values["micro_metrics"],
            "macro": values["macro_class_balanced_metrics"],
        }
        for name, values in results.items()
    }
    print(json.dumps(concise, ensure_ascii=False, sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
