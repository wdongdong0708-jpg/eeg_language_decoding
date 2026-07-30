"""Evaluate fixed-window PL shortcut baselines on one protocol partition."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.pl_speech import PLSpeechWindow, load_pl_window_jsonl
from data.pl_speech_dataset import BrainVisionSegmentReader
from evaluation.pl_shortcuts import (
    correlation_scores,
    eeg_global_field_power,
    exact_metadata_scores,
    membership_metadata_scores,
    smoothed_audio_envelope,
)
from evaluation.retrieval_metrics import metrics_from_ranks
from evaluation.speech_retrieval import (
    position_local_candidate_pools,
    ranks_from_candidate_pools,
    ranks_from_score_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-jsonl",
        default="metadata/pl_speech_windows_seed42_3s_delay_000ms.jsonl",
    )
    parser.add_argument("--partition", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoothing-ms", type=float, default=50.0)
    parser.add_argument("--position-pool-size", type=int, default=20)
    parser.add_argument(
        "--output",
        default="reports/pl_speech_shortcuts_seed42_delay000ms.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/pl_speech_shortcuts_seed42_delay000ms.md",
    )
    return parser.parse_args()


def _unique_candidates(
    windows: list[PLSpeechWindow],
) -> tuple[list[str], list[PLSpeechWindow]]:
    representatives: dict[str, PLSpeechWindow] = {}
    signatures: dict[str, tuple[object, ...]] = {}
    for window in windows:
        signature = (
            window.audio_file,
            window.audio_start_sample,
            window.audio_stop_sample,
            window.speaker_id,
            window.char_count,
        )
        previous = signatures.setdefault(window.audio_target_id, signature)
        if previous != signature:
            raise ValueError(
                f"Inconsistent candidate metadata: {window.audio_target_id}"
            )
        representatives.setdefault(window.audio_target_id, window)
    target_ids = sorted(representatives)
    return target_ids, [representatives[target_id] for target_id in target_ids]


def _audio_envelope(
    window: PLSpeechWindow,
    *,
    smoothing_ms: float,
) -> np.ndarray:
    waveform, sample_rate = sf.read(
        window.audio_file,
        start=window.audio_start_sample,
        stop=window.audio_stop_sample,
        dtype="float32",
        always_2d=True,
    )
    if int(sample_rate) != window.audio_source_sample_rate_hz:
        raise ValueError(f"Audio sample-rate mismatch: {window.audio_target_id}")
    return smoothed_audio_envelope(
        waveform,
        sample_rate_hz=int(sample_rate),
        target_time_steps=window.eeg_sample_count,
        smoothing_ms=smoothing_ms,
    )


def _metrics_both(
    scores: torch.Tensor,
    positives: list[int],
    position_pools: list[list[int]],
) -> dict[str, dict[str, float]]:
    global_ranks = ranks_from_score_matrix(
        scores,
        positives,
        tie_policy="pessimistic",
    )
    local_ranks = ranks_from_candidate_pools(
        scores,
        positives,
        position_pools,
        tie_policy="pessimistic",
    )
    return {
        "global": asdict(metrics_from_ranks(global_ranks)),
        "position_local": asdict(metrics_from_ranks(local_ranks)),
    }


def main() -> None:
    args = parse_args()
    windows = [
        window
        for window in load_pl_window_jsonl(args.window_jsonl)
        if window.split == args.partition
    ]
    if not windows:
        raise ValueError(f"No windows for partition={args.partition}")
    target_ids, candidates = _unique_candidates(windows)
    candidate_index = {
        target_id: index for index, target_id in enumerate(target_ids)
    }
    positives = [candidate_index[window.audio_target_id] for window in windows]
    candidate_positions = {
        target_id: {
            window.stimulus_position
            for window in windows
            if window.audio_target_id == target_id
        }
        for target_id in target_ids
    }
    candidate_count = len(candidates)
    query_count = len(windows)
    pool_size = min(args.position_pool_size, candidate_count)
    position_pools = position_local_candidate_pools(
        [window.stimulus_position for window in windows],
        [candidate_positions[target_id] for target_id in target_ids],
        positives,
        pool_size=pool_size,
    )

    candidate_envelopes = np.stack(
        [
            _audio_envelope(candidate, smoothing_ms=args.smoothing_ms)
            for candidate in candidates
        ]
    )
    eeg_reader = BrainVisionSegmentReader()
    query_envelopes = np.stack(
        [
            eeg_global_field_power(
                eeg_reader(
                    window.eeg_file,
                    window.eeg_start_sample,
                    window.eeg_stop_sample,
                ),
                sample_rate_hz=window.eeg_sampling_rate_hz,
                target_time_steps=window.eeg_sample_count,
                smoothing_ms=args.smoothing_ms,
            )
            for window in windows
        ]
    )
    random_generator = torch.Generator().manual_seed(args.seed)
    metadata = {
        "random": _metrics_both(
            torch.rand(
                query_count,
                candidate_count,
                generator=random_generator,
            ),
            positives,
            position_pools,
        ),
        "duration_only": _metrics_both(
            exact_metadata_scores(
                [window.window_sec for window in windows],
                [candidate.window_sec for candidate in candidates],
            ),
            positives,
            position_pools,
        ),
        "padding_mask_only": _metrics_both(
            exact_metadata_scores(
                [window.padded_eeg_samples for window in windows],
                [candidate.padded_eeg_samples for candidate in candidates],
            ),
            positives,
            position_pools,
        ),
        "character_count_only": _metrics_both(
            exact_metadata_scores(
                [window.char_count for window in windows],
                [candidate.char_count for candidate in candidates],
            ),
            positives,
            position_pools,
        ),
        "sentence_position_only": _metrics_both(
            membership_metadata_scores(
                [window.stimulus_position for window in windows],
                [candidate_positions[target_id] for target_id in target_ids],
            ),
            positives,
            position_pools,
        ),
        "subject_id_only": _metrics_both(
            exact_metadata_scores(
                [window.speaker_id for window in windows],
                [candidate.speaker_id for candidate in candidates],
            ),
            positives,
            position_pools,
        ),
        "audio_envelope": _metrics_both(
            correlation_scores(query_envelopes, candidate_envelopes),
            positives,
            position_pools,
        ),
    }
    report = {
        "schema_version": "pl-shortcut-report-v1",
        "window_jsonl": Path(args.window_jsonl).as_posix(),
        "partition": args.partition,
        "query_count": query_count,
        "candidate_count": candidate_count,
        "position_local_pool_size": pool_size,
        "tie_policy": "pessimistic",
        "audio_envelope_definition": {
            "eeg": "per-channel z-score, global field power, temporal smoothing",
            "audio": "mono RMS amplitude envelope, temporal smoothing",
            "smoothing_ms": args.smoothing_ms,
            "learned_mapping": False,
        },
        "subject_id_only_definition": (
            "known PL subject-to-speaker cohort mapping; candidates scored by speaker"
        ),
        "baselines": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = [
        "# PL speech shortcut baselines",
        "",
        f"- Partition: `{args.partition}`",
        f"- Queries: {query_count:,}",
        f"- Unique candidates: {candidate_count:,}",
        "- Tie policy: pessimistic",
        "",
        "| baseline | pool | R@1 | R@5 | R@10 | median rank | MRR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, pools in metadata.items():
        for pool_name, values in pools.items():
            markdown.append(
                f"| `{name}` | {pool_name} | {values['recall_at_1']:.4f} | "
                f"{values['recall_at_5']:.4f} | "
                f"{values['recall_at_10']:.4f} | "
                f"{values['median_rank']:.1f} | "
                f"{values['mean_reciprocal_rank']:.4f} |"
            )
    markdown.extend(
        [
            "",
            "Duration and padding are constant by construction. The subject-only "
            "control uses the known PL subject-to-speaker cohort mapping. The "
            "audio-envelope control is an untrained correlation between EEG global "
            "field power and the candidate audio RMS envelope.",
            "",
            "Sentence position is a strong shortcut in this candidate set. Formal "
            "model evaluation must report a fixed-size position-local pool in "
            "addition to the global pool.",
            "",
        ]
    )
    markdown_output = Path(args.markdown_output)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
