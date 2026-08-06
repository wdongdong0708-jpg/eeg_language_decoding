"""Typed text negatives and false-negative control for local retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

NegativeType = Literal[
    "positive_event",
    "exact_text_equivalent",
    "overlapping_local_text",
    "high_lexical_overlap",
    "semantic_similar_different_event",
    "adjacent_text",
    "same_length",
    "random",
]
FalseNegativeStrategy = Literal["mask", "soft"]


@dataclass(frozen=True, slots=True)
class TextCandidate:
    candidate_id: str
    span_event_id: str
    span_text_id: str
    global_text_id: str
    span_text: str
    span_char_count: int
    span_start_clock: int
    span_end_clock: int
    book_id: str
    stimulus_position: int
    semantic_embedding: np.ndarray | None = None

    def validate(self) -> None:
        if not self.candidate_id or not self.span_event_id or not self.span_text_id:
            raise ValueError("Candidate identifiers are required")
        if self.span_char_count <= 0 or len(self.span_text) != self.span_char_count:
            raise ValueError("span_text and span_char_count disagree")
        if self.span_end_clock - self.span_start_clock != self.span_char_count:
            raise ValueError("Clock offsets and span_char_count disagree")
        if self.semantic_embedding is not None:
            embedding = np.asarray(self.semantic_embedding)
            if embedding.ndim != 1 or not np.isfinite(embedding).all():
                raise ValueError("semantic_embedding must be a finite vector")


@dataclass(frozen=True, slots=True)
class NegativePolicyConfig:
    false_negative_strategy: FalseNegativeStrategy = "mask"
    overlap_threshold: float = 0.5
    lexical_overlap_threshold: float = 0.6
    semantic_similarity_threshold: float = 0.85
    adjacent_position_distance: int = 1
    overlap_soft_label_scale: float = 0.5
    lexical_soft_label_scale: float = 0.25

    def validate(self) -> None:
        for name in (
            "overlap_threshold",
            "lexical_overlap_threshold",
            "semantic_similarity_threshold",
            "overlap_soft_label_scale",
            "lexical_soft_label_scale",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.adjacent_position_distance < 0:
            raise ValueError("adjacent_position_distance cannot be negative")
        if self.false_negative_strategy not in {"mask", "soft"}:
            raise ValueError("Unknown false-negative strategy")


@dataclass(frozen=True, slots=True)
class PairRelationship:
    primary_type: NegativeType
    interval_overlap: float
    lexical_overlap: float
    semantic_similarity: float | None
    is_false_negative: bool


@dataclass(frozen=True, slots=True)
class PairwiseNegativePolicy:
    relationship_types: tuple[tuple[NegativeType, ...], ...]
    candidate_mask: np.ndarray
    positive_weights: np.ndarray

    def validate(self) -> None:
        if self.candidate_mask.ndim != 2:
            raise ValueError("candidate_mask must be two-dimensional")
        if self.positive_weights.shape != self.candidate_mask.shape:
            raise ValueError("positive_weights and candidate_mask shapes differ")
        if len(self.relationship_types) != self.candidate_mask.shape[0]:
            raise ValueError("relationship type row count differs")
        if any(len(row) != self.candidate_mask.shape[1] for row in self.relationship_types):
            raise ValueError("relationship type column count differs")
        if np.any(self.positive_weights < 0) or not np.isfinite(self.positive_weights).all():
            raise ValueError("positive_weights must be finite and non-negative")
        if np.any((self.positive_weights > 0) & ~self.candidate_mask):
            raise ValueError("Every positive candidate must remain unmasked")
        if np.any(self.positive_weights.sum(axis=1) <= 0):
            raise ValueError("Every anchor needs at least one positive")


def _character_ngrams(text: str, n: int = 2) -> set[str]:
    if len(text) < n:
        return set(text)
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def lexical_jaccard(left: str, right: str) -> float:
    left_items = _character_ngrams(left)
    right_items = _character_ngrams(right)
    union = left_items | right_items
    if not union:
        return 1.0
    return len(left_items & right_items) / len(union)


def interval_overlap_ratio(anchor: TextCandidate, candidate: TextCandidate) -> float:
    if anchor.global_text_id != candidate.global_text_id:
        return 0.0
    overlap = max(
        0,
        min(anchor.span_end_clock, candidate.span_end_clock)
        - max(anchor.span_start_clock, candidate.span_start_clock),
    )
    return overlap / min(anchor.span_char_count, candidate.span_char_count)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Semantic embeddings have different shapes")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def classify_candidate_pair(
    anchor: TextCandidate,
    candidate: TextCandidate,
    *,
    config: NegativePolicyConfig = NegativePolicyConfig(),
) -> PairRelationship:
    """Classify one pair with false negatives taking precedence."""

    config.validate()
    anchor.validate()
    candidate.validate()
    interval_overlap = interval_overlap_ratio(anchor, candidate)
    lexical_overlap = lexical_jaccard(anchor.span_text, candidate.span_text)
    semantic_similarity = None
    if anchor.semantic_embedding is not None and candidate.semantic_embedding is not None:
        semantic_similarity = _cosine_similarity(
            anchor.semantic_embedding,
            candidate.semantic_embedding,
        )
    if anchor.span_event_id == candidate.span_event_id:
        primary: NegativeType = "positive_event"
        false_negative = True
    elif anchor.span_text_id == candidate.span_text_id or anchor.span_text == candidate.span_text:
        primary = "exact_text_equivalent"
        false_negative = True
    elif interval_overlap >= config.overlap_threshold:
        primary = "overlapping_local_text"
        false_negative = True
    elif lexical_overlap >= config.lexical_overlap_threshold:
        primary = "high_lexical_overlap"
        false_negative = True
    elif (
        semantic_similarity is not None
        and semantic_similarity >= config.semantic_similarity_threshold
        and anchor.span_event_id != candidate.span_event_id
    ):
        primary = "semantic_similar_different_event"
        false_negative = False
    elif (
        anchor.book_id == candidate.book_id
        and abs(anchor.stimulus_position - candidate.stimulus_position)
        <= config.adjacent_position_distance
    ):
        primary = "adjacent_text"
        false_negative = False
    elif anchor.span_char_count == candidate.span_char_count:
        primary = "same_length"
        false_negative = False
    else:
        primary = "random"
        false_negative = False
    return PairRelationship(
        primary_type=primary,
        interval_overlap=interval_overlap,
        lexical_overlap=lexical_overlap,
        semantic_similarity=semantic_similarity,
        is_false_negative=false_negative,
    )


def build_pairwise_negative_policy(
    anchors: Sequence[TextCandidate],
    candidates: Sequence[TextCandidate] | None = None,
    *,
    config: NegativePolicyConfig = NegativePolicyConfig(),
) -> PairwiseNegativePolicy:
    """Build masks or soft labels for a contrastive score matrix."""

    config.validate()
    candidates = anchors if candidates is None else candidates
    if not anchors or not candidates:
        raise ValueError("anchors and candidates cannot be empty")
    mask = np.ones((len(anchors), len(candidates)), dtype=bool)
    weights = np.zeros((len(anchors), len(candidates)), dtype=np.float32)
    type_rows: list[tuple[NegativeType, ...]] = []
    for anchor_index, anchor in enumerate(anchors):
        row_types: list[NegativeType] = []
        for candidate_index, candidate in enumerate(candidates):
            relationship = classify_candidate_pair(anchor, candidate, config=config)
            row_types.append(relationship.primary_type)
            if relationship.primary_type in {"positive_event", "exact_text_equivalent"}:
                weights[anchor_index, candidate_index] = 1.0
            elif relationship.is_false_negative:
                if config.false_negative_strategy == "mask":
                    mask[anchor_index, candidate_index] = False
                elif relationship.primary_type == "overlapping_local_text":
                    weights[anchor_index, candidate_index] = (
                        relationship.interval_overlap * config.overlap_soft_label_scale
                    )
                else:
                    weights[anchor_index, candidate_index] = (
                        relationship.lexical_overlap * config.lexical_soft_label_scale
                    )
        type_rows.append(tuple(row_types))
    # A rectangular candidate set may omit the positive. Fail loudly rather than
    # silently turning the first candidate into a target.
    policy = PairwiseNegativePolicy(
        relationship_types=tuple(type_rows),
        candidate_mask=mask,
        positive_weights=weights,
    )
    policy.validate()
    return policy


def mine_typed_negatives(
    anchor: TextCandidate,
    candidates: Sequence[TextCandidate],
    *,
    quotas: Mapping[NegativeType, int],
    seed: int,
    config: NegativePolicyConfig = NegativePolicyConfig(),
) -> dict[NegativeType, tuple[int, ...]]:
    """Deterministically sample requested true-negative categories."""

    buckets: dict[NegativeType, list[int]] = {kind: [] for kind in quotas}
    for index, candidate in enumerate(candidates):
        relationship = classify_candidate_pair(anchor, candidate, config=config)
        if relationship.is_false_negative:
            continue
        if relationship.primary_type in buckets:
            buckets[relationship.primary_type].append(index)
    output: dict[NegativeType, tuple[int, ...]] = {}
    for kind, requested in quotas.items():
        if requested < 0:
            raise ValueError("Negative quotas cannot be negative")
        values = buckets[kind]
        payload = f"{seed}\0{anchor.candidate_id}\0{kind}".encode("utf-8")
        local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        generator = np.random.default_rng(local_seed)
        if len(values) > requested:
            values = [values[index] for index in generator.permutation(len(values))[:requested]]
        output[kind] = tuple(sorted(values))
    return output
