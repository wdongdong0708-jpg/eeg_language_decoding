"""Joint EEG–speech sequence retrieval model."""

from __future__ import annotations

import torch
from torch import nn

import torch.nn.functional as F

from models.losses import (
    ClipContrastiveLoss,
    DeduplicatedSigLipLoss,
    MaskedSoftTargetContrastiveLoss,
)


class EEGSpeechRetrievalModel(nn.Module):
    """Predict an unpooled speech representation at every EEG time step."""

    def __init__(
        self,
        eeg_encoder: nn.Module,
        objective: ClipContrastiveLoss,
    ) -> None:
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.objective = objective

    def forward(
        self,
        eeg: torch.Tensor,
        subject_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self.eeg_encoder, "requires_subject_indices", False):
            if subject_indices is None:
                raise ValueError("This EEG encoder requires subject_indices")
            return self.eeg_encoder(eeg, subject_indices)
        return self.eeg_encoder(eeg)

    def compute_loss(
        self,
        eeg: torch.Tensor,
        speech_targets: torch.Tensor,
        subject_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.objective(self(eeg, subject_indices), speech_targets)


class EEGTextRetrievalModel(nn.Module):
    """Map fixed-span EEG and frozen text targets into one pooled space."""

    def __init__(
        self,
        *,
        eeg_encoder: nn.Module,
        eeg_projection: nn.Module,
        objective: MaskedSoftTargetContrastiveLoss | DeduplicatedSigLipLoss,
        text_projection: nn.Module | None = None,
        text_sequence_pooler: nn.Module | None = None,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.eeg_projection = eeg_projection
        self.objective = objective
        self.text_projection = text_projection or nn.Identity()
        self.text_sequence_pooler = text_sequence_pooler
        self.normalize_embeddings = bool(normalize_embeddings)

    def encode_eeg(
        self,
        eeg: torch.Tensor,
        subject_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self.eeg_encoder, "requires_subject_indices", False):
            if subject_indices is None:
                raise ValueError("This EEG encoder requires subject_indices")
            encoded = self.eeg_encoder(eeg, subject_indices)
        else:
            encoded = self.eeg_encoder(eeg)
        projected = self.eeg_projection(encoded)
        if self.normalize_embeddings:
            projected = F.normalize(projected, dim=1)
        return projected

    def encode_text(self, text_targets: torch.Tensor) -> torch.Tensor:
        if text_targets.ndim == 3:
            if self.text_sequence_pooler is None:
                text_targets = text_targets.mean(dim=1)
            else:
                text_targets = self.text_sequence_pooler(text_targets)
        if text_targets.ndim != 2:
            raise ValueError("Text targets must be [B,D] or [B,L,D]")
        projected = self.text_projection(text_targets)
        if self.normalize_embeddings:
            projected = F.normalize(projected, dim=1)
        return projected

    def forward(
        self,
        eeg: torch.Tensor,
        text_targets: torch.Tensor,
        subject_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.encode_eeg(eeg, subject_indices),
            self.encode_text(text_targets),
        )

    def compute_loss(
        self,
        eeg: torch.Tensor,
        text_targets: torch.Tensor,
        *,
        subject_indices: torch.Tensor | None = None,
        positive_weights: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        estimates, candidates = self(eeg, text_targets, subject_indices)
        return self.objective(
            estimates,
            candidates,
            positive_weights=positive_weights,
            candidate_mask=candidate_mask,
        )
