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

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.eeg_encoder(eeg)

    def compute_loss(
        self,
        eeg: torch.Tensor,
        speech_targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.objective(self(eeg), speech_targets)
