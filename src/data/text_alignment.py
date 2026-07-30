"""Auditable monotonic alignment of stimulus-row text sequences.

This module aligns sequences, never independent row numbers. Exact normalized
anchors are found first. Fuzzy matches are only accepted inside the monotonic
gaps between those anchors and must be reciprocal best matches.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from rapidfuzz.fuzz import ratio

ALIGNMENT_VERSION = "monotonic-text-alignment-v1"
OVERRIDE_COLUMNS = (
    "source_sentence_id",
    "target_sentence_id",
    "target_global_text_id",
    "decision",
    "score",
    "reviewer",
    "note",
)


@dataclass(frozen=True, slots=True)
class AlignmentTextUnit:
    sentence_id: str
    raw_text: str
    normalized_text: str
    global_text_id: str | None = None


@dataclass(frozen=True, slots=True)
class TextAlignment:
    source_sentence_id: str
    target_sentence_id: str | None
    global_text_id: str | None
    status: str
    score: float | None
    evidence: str


def ensure_override_file(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=OVERRIDE_COLUMNS).writeheader()


def load_overrides(path: str | Path) -> dict[str, dict[str, str]]:
    override_path = Path(path)
    if not override_path.is_file():
        return {}
    with override_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OVERRIDE_COLUMNS:
            raise ValueError(
                f"Override schema mismatch in {override_path}: {reader.fieldnames}"
            )
        rows = {row["source_sentence_id"]: row for row in reader}
    if "" in rows:
        raise ValueError("Manual override source_sentence_id cannot be empty")
    return rows


def _reciprocal_fuzzy_pairs(
    source: list[AlignmentTextUnit],
    target: list[AlignmentTextUnit],
    *,
    minimum_score: float,
) -> list[tuple[int, int, float]]:
    if not source or not target:
        return []
    scores: dict[tuple[int, int], float] = {}
    for source_index, source_unit in enumerate(source):
        source_text = source_unit.normalized_text
        if not source_text:
            continue
        for target_index, target_unit in enumerate(target):
            target_text = target_unit.normalized_text
            if not target_text:
                continue
            maximum_length = max(len(source_text), len(target_text))
            if maximum_length and abs(len(source_text) - len(target_text)) > max(
                2, round(maximum_length * 0.15)
            ):
                continue
            value = float(ratio(source_text, target_text))
            if value >= minimum_score:
                scores[(source_index, target_index)] = value

    best_target: dict[int, tuple[int, float]] = {}
    best_source: dict[int, tuple[int, float]] = {}
    for (source_index, target_index), value in scores.items():
        current_target = best_target.get(source_index)
        candidate_target = (target_index, value)
        if current_target is None or (value, -target_index) > (
            current_target[1],
            -current_target[0],
        ):
            best_target[source_index] = candidate_target
        current_source = best_source.get(target_index)
        candidate_source = (source_index, value)
        if current_source is None or (value, -source_index) > (
            current_source[1],
            -current_source[0],
        ):
            best_source[target_index] = candidate_source

    pairs = [
        (source_index, target_index, value)
        for source_index, (target_index, value) in best_target.items()
        if best_source.get(target_index, (-1, 0.0))[0] == source_index
    ]
    pairs.sort()
    monotonic: list[tuple[int, int, float]] = []
    previous_target = -1
    for pair in pairs:
        if pair[1] > previous_target:
            monotonic.append(pair)
            previous_target = pair[1]
    return monotonic


def align_monotonic_text_sequences(
    source: list[AlignmentTextUnit],
    target: list[AlignmentTextUnit],
    *,
    overrides: dict[str, dict[str, str]] | None = None,
    minimum_fuzzy_score: float = 96.0,
) -> dict[str, TextAlignment]:
    """Align source to target using exact anchors plus conservative fuzzy gaps."""

    overrides = overrides or {}
    target_by_sentence = {unit.sentence_id: unit for unit in target}
    target_by_global = {
        unit.global_text_id: unit for unit in target if unit.global_text_id is not None
    }
    results: dict[str, TextAlignment] = {}

    matcher = SequenceMatcher(
        None,
        [unit.normalized_text for unit in source],
        [unit.normalized_text for unit in target],
        autojunk=False,
    )
    exact_pairs: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            source_index = block.a + offset
            target_index = block.b + offset
            source_unit = source[source_index]
            target_unit = target[target_index]
            status = (
                "exact"
                if source_unit.raw_text == target_unit.raw_text
                else "normalized"
            )
            results[source_unit.sentence_id] = TextAlignment(
                source_sentence_id=source_unit.sentence_id,
                target_sentence_id=target_unit.sentence_id,
                global_text_id=target_unit.global_text_id,
                status=status,
                score=100.0,
                evidence=f"{ALIGNMENT_VERSION}:sequence_matcher_anchor",
            )
            exact_pairs.append((source_index, target_index))

    anchors = [(-1, -1), *exact_pairs, (len(source), len(target))]
    for (source_left, target_left), (source_right, target_right) in zip(
        anchors, anchors[1:]
    ):
        if source_right == source_left + 1 or target_right == target_left + 1:
            continue
        source_gap = source[source_left + 1 : source_right]
        target_gap = target[target_left + 1 : target_right]
        for local_source, local_target, score in _reciprocal_fuzzy_pairs(
            source_gap,
            target_gap,
            minimum_score=minimum_fuzzy_score,
        ):
            source_unit = source_gap[local_source]
            target_unit = target_gap[local_target]
            results[source_unit.sentence_id] = TextAlignment(
                source_sentence_id=source_unit.sentence_id,
                target_sentence_id=target_unit.sentence_id,
                global_text_id=target_unit.global_text_id,
                status="fuzzy",
                score=score,
                evidence=(
                    f"{ALIGNMENT_VERSION}:reciprocal_best_within_exact_anchor_gap"
                ),
            )

    for source_unit in source:
        override = overrides.get(source_unit.sentence_id)
        if override is not None:
            target_unit = None
            target_sentence_id = override.get("target_sentence_id", "")
            target_global_text_id = override.get("target_global_text_id", "")
            if target_sentence_id:
                target_unit = target_by_sentence.get(target_sentence_id)
            elif target_global_text_id:
                target_unit = target_by_global.get(target_global_text_id)
            decision = override.get("decision", "").strip().casefold()
            if decision == "match" and target_unit is not None:
                results[source_unit.sentence_id] = TextAlignment(
                    source_sentence_id=source_unit.sentence_id,
                    target_sentence_id=target_unit.sentence_id,
                    global_text_id=target_unit.global_text_id,
                    status="manual",
                    score=float(override["score"]) if override.get("score") else 100.0,
                    evidence=f"{ALIGNMENT_VERSION}:manual_override",
                )
            elif decision == "unresolved":
                results.pop(source_unit.sentence_id, None)
            else:
                raise ValueError(
                    f"Invalid or dangling override for {source_unit.sentence_id}"
                )

        if source_unit.sentence_id not in results:
            results[source_unit.sentence_id] = TextAlignment(
                source_sentence_id=source_unit.sentence_id,
                target_sentence_id=None,
                global_text_id=None,
                status="unresolved",
                score=None,
                evidence=f"{ALIGNMENT_VERSION}:no_accepted_match",
            )
    return results

