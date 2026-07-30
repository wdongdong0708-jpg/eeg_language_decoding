"""Sentence-, token- and character-level text representation extraction."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from data.alignment import is_highlighted_character

OFFICIAL_TEXT_MODEL = "bert-base-chinese"
OFFICIAL_TEXT_POOLING = "last_hidden_state.mean(dim=1)"

SentencePooling = Literal["mean_content_tokens", "mean_attended_tokens", "cls"]


@dataclass(frozen=True, slots=True)
class TextFeatureInput:
    content_id: str
    text: str

    def validate(self) -> None:
        if not self.content_id:
            raise ValueError("content_id is required")
        if not self.text or not self.text.strip():
            raise ValueError("text must contain at least one non-whitespace character")


@dataclass(frozen=True, slots=True)
class TextFeatureConfig:
    model_id: str = OFFICIAL_TEXT_MODEL
    layer_index: int = -1
    sentence_pooling: SentencePooling = "mean_content_tokens"
    max_length: int = 512
    batch_size: int = 16
    output_dtype: str = "float32"
    strict_character_alignment: bool = True

    def validate(self) -> None:
        if self.max_length < 3:
            raise ValueError("max_length must leave room for content and special tokens")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.sentence_pooling not in {
            "mean_content_tokens",
            "mean_attended_tokens",
            "cls",
        }:
            raise ValueError(f"Unknown sentence pooling: {self.sentence_pooling}")
        np.dtype(self.output_dtype)


@dataclass(frozen=True, slots=True)
class TextFeatureResult:
    content_id: str
    text: str
    model_id: str
    layer_index: int
    sentence_pooling: str
    sentence_hidden_state: np.ndarray
    token_indices: np.ndarray
    token_ids: np.ndarray
    tokens: tuple[str, ...]
    token_offsets: np.ndarray
    special_tokens_mask: np.ndarray
    token_hidden_states: np.ndarray
    character_indices: np.ndarray
    characters: tuple[str, ...]
    character_offsets: np.ndarray
    character_hidden_states: np.ndarray
    character_is_highlighted: np.ndarray
    character_token_indices: tuple[tuple[int, ...], ...]
    truncated: bool
    unmapped_character_indices: tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.content_id or not self.text:
            raise ValueError("Text feature provenance is incomplete")
        token_count = len(self.tokens)
        character_count = len(self.characters)
        hidden_size = int(self.sentence_hidden_state.shape[0])
        if self.sentence_hidden_state.ndim != 1:
            raise ValueError("sentence_hidden_state must have shape [hidden]")
        if self.token_hidden_states.shape != (token_count, hidden_size):
            raise ValueError("token_hidden_states shape is inconsistent")
        if self.token_offsets.shape != (token_count, 2):
            raise ValueError("token_offsets must have shape [tokens, 2]")
        if self.token_indices.shape != (token_count,):
            raise ValueError("token_indices must have shape [tokens]")
        if self.token_ids.shape != (token_count,):
            raise ValueError("token_ids must have shape [tokens]")
        if self.special_tokens_mask.shape != (token_count,):
            raise ValueError("special_tokens_mask must have shape [tokens]")
        if self.character_hidden_states.shape != (character_count, hidden_size):
            raise ValueError("character_hidden_states shape is inconsistent")
        if self.character_offsets.shape != (character_count, 2):
            raise ValueError("character_offsets must have shape [characters, 2]")
        if self.character_indices.shape != (character_count,):
            raise ValueError("character_indices must have shape [characters]")
        if self.character_is_highlighted.shape != (character_count,):
            raise ValueError("character_is_highlighted must have shape [characters]")
        if len(self.character_token_indices) != character_count:
            raise ValueError("character_token_indices length is inconsistent")


def _sentence_pool(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    special_tokens_mask: np.ndarray,
    offsets: np.ndarray,
    pooling: SentencePooling,
) -> np.ndarray:
    attended = attention_mask.astype(bool)
    if pooling == "cls":
        attended_indices = np.flatnonzero(attended)
        if not len(attended_indices):
            raise ValueError("Cannot CLS-pool an empty sequence")
        return hidden_states[attended_indices[0]]
    if pooling == "mean_attended_tokens":
        selected = attended
    elif pooling == "mean_content_tokens":
        selected = (
            attended
            & ~special_tokens_mask.astype(bool)
            & (offsets[:, 1] > offsets[:, 0])
        )
    else:
        raise ValueError(f"Unknown sentence pooling: {pooling}")
    if not selected.any():
        raise ValueError(f"No tokens available for pooling={pooling}")
    return hidden_states[selected].mean(axis=0)


def assemble_text_features(
    *,
    item: TextFeatureInput,
    model_id: str,
    layer_index: int,
    sentence_pooling: SentencePooling,
    token_ids: np.ndarray,
    tokens: Sequence[str],
    offsets: np.ndarray,
    attention_mask: np.ndarray,
    special_tokens_mask: np.ndarray,
    hidden_states: np.ndarray,
    truncated: bool,
    output_dtype: str = "float32",
    strict_character_alignment: bool = True,
) -> TextFeatureResult:
    """Assemble all granularities from tokenizer offsets and model states."""

    item.validate()
    token_ids = np.asarray(token_ids)
    offsets = np.asarray(offsets, dtype=np.int64)
    attention_mask = np.asarray(attention_mask, dtype=bool)
    special_tokens_mask = np.asarray(special_tokens_mask, dtype=bool)
    hidden_states = np.asarray(hidden_states)
    sequence_length = token_ids.shape[0]
    expected_vector_shapes = {
        "offsets": offsets.shape[0],
        "attention_mask": attention_mask.shape[0],
        "special_tokens_mask": special_tokens_mask.shape[0],
        "hidden_states": hidden_states.shape[0],
        "tokens": len(tokens),
    }
    if any(length != sequence_length for length in expected_vector_shapes.values()):
        raise ValueError(
            f"Inconsistent token sequence lengths: {expected_vector_shapes}"
        )
    if offsets.shape != (sequence_length, 2) or hidden_states.ndim != 2:
        raise ValueError("Expected offsets [tokens,2] and hidden_states [tokens,hidden]")

    sentence_hidden = _sentence_pool(
        hidden_states,
        attention_mask,
        special_tokens_mask,
        offsets,
        sentence_pooling,
    )
    kept_positions = np.flatnonzero(attention_mask)
    kept_offsets = offsets[kept_positions]
    kept_special = special_tokens_mask[kept_positions]
    kept_hidden = hidden_states[kept_positions]
    kept_ids = token_ids[kept_positions]
    kept_tokens = tuple(tokens[index] for index in kept_positions)

    character_indices: list[int] = []
    characters: list[str] = []
    character_offsets: list[tuple[int, int]] = []
    character_hidden: list[np.ndarray] = []
    character_highlighted: list[bool] = []
    character_tokens: list[tuple[int, ...]] = []
    unmapped: list[int] = []
    for character_index, character in enumerate(item.text):
        if character.isspace():
            continue
        overlapping_local_positions = [
            local_position
            for local_position, (start, stop) in enumerate(kept_offsets)
            if not kept_special[local_position]
            and int(start) < character_index + 1
            and int(stop) > character_index
        ]
        if not overlapping_local_positions:
            unmapped.append(character_index)
            continue
        source_token_indices = tuple(
            int(kept_positions[local_position])
            for local_position in overlapping_local_positions
        )
        character_indices.append(character_index)
        characters.append(character)
        character_offsets.append((character_index, character_index + 1))
        character_hidden.append(
            kept_hidden[overlapping_local_positions].mean(axis=0)
        )
        character_highlighted.append(is_highlighted_character(character))
        character_tokens.append(source_token_indices)

    if unmapped and strict_character_alignment:
        raise ValueError(
            f"Tokenizer offsets did not cover character indices {unmapped} "
            f"for content_id={item.content_id}"
        )

    dtype = np.dtype(output_dtype)
    hidden_size = hidden_states.shape[1]
    result = TextFeatureResult(
        content_id=item.content_id,
        text=item.text,
        model_id=model_id,
        layer_index=layer_index,
        sentence_pooling=sentence_pooling,
        sentence_hidden_state=np.asarray(sentence_hidden, dtype=dtype),
        token_indices=np.asarray(kept_positions, dtype=np.int64),
        token_ids=np.asarray(kept_ids, dtype=np.int64),
        tokens=kept_tokens,
        token_offsets=np.asarray(kept_offsets, dtype=np.int64),
        special_tokens_mask=np.asarray(kept_special, dtype=bool),
        token_hidden_states=np.asarray(kept_hidden, dtype=dtype),
        character_indices=np.asarray(character_indices, dtype=np.int64),
        characters=tuple(characters),
        character_offsets=np.asarray(character_offsets, dtype=np.int64).reshape(-1, 2),
        character_hidden_states=(
            np.asarray(character_hidden, dtype=dtype).reshape(-1, hidden_size)
        ),
        character_is_highlighted=np.asarray(character_highlighted, dtype=bool),
        character_token_indices=tuple(character_tokens),
        truncated=truncated,
        unmapped_character_indices=tuple(unmapped),
    )
    result.validate()
    return result


class TextEmbeddingExtractor:
    """Hugging Face extractor with padding-safe sentence pooling."""

    def __init__(
        self,
        *,
        tokenizer: object,
        model: object,
        config: TextFeatureConfig,
        device: str | None = None,
    ) -> None:
        config.validate()
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for character offsets")
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        config: TextFeatureConfig,
        *,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> "TextEmbeddingExtractor":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            use_fast=True,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            config.model_id,
            local_files_only=local_files_only,
        )
        if device is not None:
            model = model.to(device)
        model.eval()
        return cls(tokenizer=tokenizer, model=model, config=config, device=device)

    def extract(self, items: Sequence[TextFeatureInput]) -> list[TextFeatureResult]:
        import torch

        results: list[TextFeatureResult] = []
        for batch_start in range(0, len(items), self.config.batch_size):
            batch = list(items[batch_start : batch_start + self.config.batch_size])
            for item in batch:
                item.validate()
            texts = [item.text for item in batch]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
                return_attention_mask=True,
                return_special_tokens_mask=True,
            )
            offsets = encoded.pop("offset_mapping")
            special_tokens_mask = encoded.pop("special_tokens_mask")
            model_inputs = {
                name: tensor.to(self.device) if self.device else tensor
                for name, tensor in encoded.items()
            }
            needs_all_layers = self.config.layer_index != -1
            with torch.inference_mode():
                outputs = self.model(
                    **model_inputs,
                    output_hidden_states=needs_all_layers,
                )
            if self.config.layer_index == -1:
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[self.config.layer_index]

            hidden_np = hidden.detach().cpu().numpy()
            ids_np = encoded["input_ids"].detach().cpu().numpy()
            attention_np = encoded["attention_mask"].detach().cpu().numpy()
            offsets_np = offsets.detach().cpu().numpy()
            special_np = special_tokens_mask.detach().cpu().numpy()
            for batch_index, item in enumerate(batch):
                full_ids = self.tokenizer(
                    item.text,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
                tokens = self.tokenizer.convert_ids_to_tokens(
                    ids_np[batch_index].tolist()
                )
                results.append(
                    assemble_text_features(
                        item=item,
                        model_id=self.config.model_id,
                        layer_index=self.config.layer_index,
                        sentence_pooling=self.config.sentence_pooling,
                        token_ids=ids_np[batch_index],
                        tokens=tokens,
                        offsets=offsets_np[batch_index],
                        attention_mask=attention_np[batch_index],
                        special_tokens_mask=special_np[batch_index],
                        hidden_states=hidden_np[batch_index],
                        truncated=len(full_ids) > self.config.max_length,
                        output_dtype=self.config.output_dtype,
                        strict_character_alignment=(
                            self.config.strict_character_alignment
                        ),
                    )
                )
        return results


def _character_token_csr(
    mappings: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    indptr = [0]
    values: list[int] = []
    for mapping in mappings:
        values.extend(mapping)
        indptr.append(len(values))
    return np.asarray(indptr, dtype=np.int64), np.asarray(values, dtype=np.int64)


def save_text_features(path: str | Path, result: TextFeatureResult) -> None:
    """Save a no-pickle NPZ with hidden states and complete offset provenance."""

    result.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    indptr, token_values = _character_token_csr(result.character_token_indices)
    metadata = {
        "content_id": result.content_id,
        "text": result.text,
        "model_id": result.model_id,
        "layer_index": result.layer_index,
        "sentence_pooling": result.sentence_pooling,
        "truncated": result.truncated,
        "unmapped_character_indices": list(result.unmapped_character_indices),
    }
    handle = tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=target.stem + ".",
        suffix=".npz",
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            sentence_hidden_state=result.sentence_hidden_state,
            token_indices=result.token_indices,
            token_ids=result.token_ids,
            tokens=np.asarray(result.tokens),
            token_offsets=result.token_offsets,
            special_tokens_mask=result.special_tokens_mask,
            token_hidden_states=result.token_hidden_states,
            character_indices=result.character_indices,
            characters=np.asarray(result.characters),
            character_offsets=result.character_offsets,
            character_hidden_states=result.character_hidden_states,
            character_is_highlighted=result.character_is_highlighted,
            character_token_indptr=indptr,
            character_token_indices=token_values,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_text_features(path: str | Path) -> TextFeatureResult:
    """Load a feature file without enabling NumPy pickle deserialization."""

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        indptr = archive["character_token_indptr"].astype(np.int64)
        token_values = archive["character_token_indices"].astype(np.int64)
        mappings = tuple(
            tuple(int(value) for value in token_values[indptr[index] : indptr[index + 1]])
            for index in range(len(indptr) - 1)
        )
        result = TextFeatureResult(
            content_id=metadata["content_id"],
            text=metadata["text"],
            model_id=metadata["model_id"],
            layer_index=int(metadata["layer_index"]),
            sentence_pooling=metadata["sentence_pooling"],
            sentence_hidden_state=archive["sentence_hidden_state"],
            token_indices=archive["token_indices"],
            token_ids=archive["token_ids"],
            tokens=tuple(str(value) for value in archive["tokens"]),
            token_offsets=archive["token_offsets"],
            special_tokens_mask=archive["special_tokens_mask"],
            token_hidden_states=archive["token_hidden_states"],
            character_indices=archive["character_indices"],
            characters=tuple(str(value) for value in archive["characters"]),
            character_offsets=archive["character_offsets"],
            character_hidden_states=archive["character_hidden_states"],
            character_is_highlighted=archive["character_is_highlighted"],
            character_token_indices=mappings,
            truncated=bool(metadata["truncated"]),
            unmapped_character_indices=tuple(
                int(value) for value in metadata["unmapped_character_indices"]
            ),
        )
    result.validate()
    return result
