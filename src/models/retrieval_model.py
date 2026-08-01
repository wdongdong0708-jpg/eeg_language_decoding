"""Joint EEG–speech sequence retrieval model."""

from __future__ import annotations

import torch
from torch import nn

from models.losses import ClipContrastiveLoss


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
