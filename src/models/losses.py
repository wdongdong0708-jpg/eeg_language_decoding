"""Contrastive objectives for unpooled sequence representations."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

NormKind = Literal["none", "x", "y", "xy"]


def sequence_similarity(
    estimates: torch.Tensor,
    candidates: torch.Tensor,
    *,
    norm_kind: NormKind = "y",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise similarity after flattening channel and time dimensions.

    Args:
        estimates: Predicted features, shape ``[B, C, T]``.
        candidates: Target features, shape ``[N, C, T]``.
        norm_kind:
            ``"none"``: raw dot product over C*T.
            ``"x"``: normalize estimates only.
            ``"y"``: normalize candidates only, matching brainmagick's default.
            ``"xy"``: cosine similarity over flattened C*T.

    Returns:
        Score matrix of shape ``[B, N]``.
    """

    if estimates.ndim != 3 or candidates.ndim != 3:
        raise ValueError(
            "estimates and candidates must have shape [batch, channels, time]."
        )
    if estimates.shape[1:] != candidates.shape[1:]:
        raise ValueError(
            "estimates and candidates must share [channels, time], got "
            f"{tuple(estimates.shape[1:])} and {tuple(candidates.shape[1:])}."
        )

    x = estimates.reshape(estimates.shape[0], -1)
    y = candidates.reshape(candidates.shape[0], -1)
    scores = x @ y.T

    if norm_kind == "none":
        return scores
    if norm_kind == "x":
        return scores / x.norm(dim=1, keepdim=True).clamp_min(eps)
    if norm_kind == "y":
        return scores / y.norm(dim=1).clamp_min(eps).unsqueeze(0)
    if norm_kind == "xy":
        x = F.normalize(x, dim=1, eps=eps)
        y = F.normalize(y, dim=1, eps=eps)
        return x @ y.T
    raise ValueError(
        f"Invalid norm_kind={norm_kind!r}. Expected none, x, y, or xy."
    )


class ClipContrastiveLoss(nn.Module):
    """CLIP-style classification loss with optional symmetric direction."""

    def __init__(
        self,
        *,
        norm_kind: NormKind = "y",
        learn_temperature: bool = False,
        temperature: float = 0.07,
        symmetric: bool = False,
        center: bool = False,
        pool: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.norm_kind = norm_kind
        self.symmetric = symmetric
        self.center = center
        self.pool = pool
        self.eps = eps

        logit_scale = torch.tensor(1.0 / temperature).log()
        self.logit_scale = nn.Parameter(
            logit_scale,
            requires_grad=learn_temperature,
        )

    def _prepare(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.center:
            estimates = estimates - estimates.mean(dim=(1, 2), keepdim=True)
            candidates = candidates - candidates.mean(dim=(1, 2), keepdim=True)
        if self.pool:
            estimates = estimates.mean(dim=2, keepdim=True)
            candidates = candidates.mean(dim=2, keepdim=True)
        return estimates, candidates

    def get_scores(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        estimates, candidates = self._prepare(estimates, candidates)
        scores = sequence_similarity(
            estimates,
            candidates,
            norm_kind=self.norm_kind,
            eps=self.eps,
        )
        return scores * self.logit_scale.exp()

    def get_probabilities(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return self.get_scores(estimates, candidates).softmax(dim=1)

    def forward(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if estimates.shape[0] > candidates.shape[0]:
            raise ValueError(
                "candidates must contain at least one target per estimate."
            )

        scores = self.get_scores(estimates, candidates)
        targets = torch.arange(estimates.shape[0], device=estimates.device)
        loss = F.cross_entropy(scores, targets)

        if self.symmetric:
            if scores.shape[0] != scores.shape[1]:
                raise ValueError(
                    "symmetric=True requires no extra negatives, so scores is square."
                )
            loss = 0.5 * (loss + F.cross_entropy(scores.T, targets))
        return loss
