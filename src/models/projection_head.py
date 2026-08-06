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
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim) <= 0:
            raise ValueError("Projection dimensions must be positive")
        if sequence_axis not in {"last", "middle"}:
            raise ValueError("sequence_axis must be last (EEG) or middle (text)")
        if pooling not in {"mean", "attention"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.sequence_axis = sequence_axis
        self.pooling = pooling
        self.attention = AttentionPool1d(input_dim) if pooling == "attention" else None
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
        else:
            assert self.attention is not None
            pooled = self.attention(sequence)
        return self.projection(pooled)
