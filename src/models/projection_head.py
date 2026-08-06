"""Projection and attention-pooling heads for a shared retrieval space."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPool1d(nn.Module):
    """Learned scalar attention over an already fixed-length sequence."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.query = nn.Parameter(torch.empty(feature_dim))
        nn.init.normal_(self.query, std=feature_dim**-0.5)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("Expected sequence [batch, length, features]")
        if sequence.shape[2] != self.query.shape[0]:
            raise ValueError("Sequence feature dimension differs from attention query")
        logits = torch.einsum("blf,f->bl", sequence, self.query)
        weights = logits.softmax(dim=1)
        return torch.einsum("bl,blf->bf", weights, sequence)


class BahdanauAttention(nn.Module):
    """Additive temporal attention over keys shaped ``[B, F, T]``.

    This follows the Bahdanau pooling interface used by the NeuroAI
    ``SimpleConvTimeAgg`` model.  Optional queries are retained for parity
    with that implementation, although EEG temporal pooling uses keys only.
    """

    def __init__(self, input_size: int | None, hidden_size: int) -> None:
        super().__init__()
        if input_size is not None and input_size <= 0:
            raise ValueError("input_size must be positive or None")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if input_size is None:
            self.Wa = nn.LazyLinear(hidden_size)
            self.Ua = nn.LazyLinear(hidden_size)
        else:
            self.Wa = nn.Linear(input_size, hidden_size)
            self.Ua = nn.Linear(input_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def attention_weights(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized temporal weights shaped ``[B, 1, T]``."""
        if keys.ndim != 3:
            raise ValueError("Expected keys [batch, features, time]")
        keys_by_time = keys.transpose(2, 1)
        summed = self.Wa(keys_by_time)
        if queries is not None:
            if queries.shape != keys.shape:
                raise ValueError("queries must have the same shape as keys")
            summed = summed + self.Ua(queries.transpose(2, 1))
        scores = self.Va(torch.tanh(summed))
        return torch.nn.functional.softmax(scores.squeeze(2).unsqueeze(1), dim=-1)

    def forward(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool ``[B, F, T]`` keys into a ``[B, F, 1]`` context."""
        weights = self.attention_weights(keys, queries)
        context = torch.bmm(weights, keys.transpose(2, 1))
        return context.transpose(2, 1)


class PooledProjectionHead(nn.Module):
    """Pool EEG [B,C,T] or text [B,L,C], then project and normalize."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        sequence_axis: str,
        pooling: str = "mean",
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bahdanau_attention_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim) <= 0:
            raise ValueError("Projection dimensions must be positive")
        if sequence_axis not in {"last", "middle"}:
            raise ValueError("sequence_axis must be last (EEG) or middle (text)")
        if pooling not in {
            "mean",
            "attention",
            "bahdanau_attention",
        }:
            raise ValueError(f"Unknown pooling: {pooling}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.sequence_axis = sequence_axis
        self.pooling = pooling
        if pooling == "attention":
            self.attention: nn.Module | None = AttentionPool1d(input_dim)
        elif pooling == "bahdanau_attention":
            self.attention = BahdanauAttention(
                input_size=input_dim,
                hidden_size=bahdanau_attention_hidden_size,
            )
        else:
            self.attention = None
        hidden_dim = output_dim if hidden_dim is None else hidden_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("Expected a rank-3 sequence tensor")
        if self.sequence_axis == "last":
            sequence = sequence.transpose(1, 2)
        if self.pooling == "mean":
            pooled = sequence.mean(dim=1)
        elif self.pooling == "bahdanau_attention":
            assert self.attention is not None
            pooled = self.attention(sequence.transpose(1, 2)).squeeze(-1)
        else:
            assert self.attention is not None
            pooled = self.attention(sequence)
        return self.projection(pooled)
