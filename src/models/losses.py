"""Contrastive objectives for unpooled sequence representations."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

NormKind = Literal["none", "x", "y", "xy"]


def embedding_similarity(
    estimates: torch.Tensor,
    candidates: torch.Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Similarity for pooled embeddings of shape ``[B, D]``."""

    if estimates.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Pooled embeddings must have shape [batch, dimension]")
    if estimates.shape[1] != candidates.shape[1]:
        raise ValueError("Estimate and candidate dimensions differ")
    if normalize:
        estimates = F.normalize(estimates, dim=1, eps=eps)
        candidates = F.normalize(candidates, dim=1, eps=eps)
    return estimates @ candidates.T


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


class MaskedSoftTargetContrastiveLoss(nn.Module):
    """Pooled InfoNCE with explicit false-negative masks or soft positives."""

    def __init__(
        self,
        *,
        learn_temperature: bool = False,
        temperature: float = 0.07,
        symmetric: bool = True,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.symmetric = symmetric
        self.normalize = normalize
        self.eps = eps
        scale = torch.tensor(1.0 / temperature).log()
        self.logit_scale = nn.Parameter(scale, requires_grad=learn_temperature)

    def get_scores(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return embedding_similarity(
            estimates,
            candidates,
            normalize=self.normalize,
            eps=self.eps,
        ) * self.logit_scale.exp()

    def forward(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
        *,
        positive_weights: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.get_scores(estimates, candidates)
        if positive_weights is None:
            if scores.shape[0] > scores.shape[1]:
                raise ValueError("Candidate set omits at least one diagonal positive")
            positive_weights = torch.zeros_like(scores)
            diagonal = torch.arange(scores.shape[0], device=scores.device)
            positive_weights[diagonal, diagonal] = 1.0
        else:
            positive_weights = positive_weights.to(
                device=scores.device,
                dtype=scores.dtype,
            )
        if candidate_mask is None:
            candidate_mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            candidate_mask = candidate_mask.to(device=scores.device, dtype=torch.bool)
        loss = self._directional_loss(scores, positive_weights, candidate_mask)
        if self.symmetric:
            if scores.shape[0] != scores.shape[1]:
                raise ValueError("symmetric=True requires a square score matrix")
            reverse = self._directional_loss(
                scores.T,
                positive_weights.T,
                candidate_mask.T,
            )
            loss = 0.5 * (loss + reverse)
        return loss

    @staticmethod
    def _directional_loss(
        scores: torch.Tensor,
        positive_weights: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if positive_weights.shape != scores.shape or candidate_mask.shape != scores.shape:
            raise ValueError("Policy tensors must have the same shape as scores")
        if torch.any(positive_weights < 0) or not torch.isfinite(positive_weights).all():
            raise ValueError("positive_weights must be finite and non-negative")
        if torch.any((positive_weights > 0) & ~candidate_mask):
            raise ValueError("A positive candidate was masked")
        positive_mass = positive_weights.sum(dim=1, keepdim=True)
        if torch.any(positive_mass <= 0):
            raise ValueError("Every query requires positive weight")
        if torch.any(candidate_mask.sum(dim=1) == 0):
            raise ValueError("Every query requires an unmasked candidate")
        targets = positive_weights / positive_mass
        masked_scores = scores.masked_fill(
            ~candidate_mask,
            torch.finfo(scores.dtype).min,
        )
        log_probabilities = F.log_softmax(masked_scores, dim=1)
        return -(targets * log_probabilities).sum(dim=1).mean()


class DeduplicatedSigLipLoss(nn.Module):
    """D-SigLIP for pooled embeddings with duplicate and false-negative masks.

    Every EEG occurrence remains an anchor and keeps its diagonal text target.
    Off-diagonal candidates with the same frozen text target are excluded from
    the binary loss instead of being counted as negatives or repeated positive
    candidates.  The reduction follows the reference implementation: sum all
    retained pair losses and divide by the number of EEG anchors.
    """

    def __init__(
        self,
        *,
        logit_scale_init: float = 10.0,
        bias_init: float = -10.0,
        learn_temperature: bool = True,
        learn_bias: bool = True,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if logit_scale_init <= 0:
            raise ValueError("logit_scale_init must be positive")
        self.normalize = bool(normalize)
        self.eps = float(eps)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(logit_scale_init)).log(),
            requires_grad=learn_temperature,
        )
        self.bias = nn.Parameter(
            torch.tensor(float(bias_init)), requires_grad=learn_bias
        )

    def get_scores(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        similarities = embedding_similarity(
            estimates,
            candidates,
            normalize=self.normalize,
            eps=self.eps,
        )
        return similarities * self.logit_scale.exp() + self.bias

    def forward(
        self,
        estimates: torch.Tensor,
        candidates: torch.Tensor,
        *,
        positive_weights: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.get_scores(estimates, candidates)
        if scores.shape[0] != scores.shape[1]:
            raise ValueError(
                "D-SigLIP expects one original candidate per EEG occurrence"
            )
        count = scores.shape[0]
        diagonal = torch.eye(count, device=scores.device, dtype=torch.bool)
        if positive_weights is None:
            equivalent = diagonal
        else:
            weights = positive_weights.to(device=scores.device, dtype=scores.dtype)
            if weights.shape != scores.shape:
                raise ValueError("positive_weights must have the score-matrix shape")
            if not torch.isfinite(weights).all() or torch.any(weights < 0):
                raise ValueError("positive_weights must be finite and non-negative")
            if torch.any((weights != 0) & (weights != 1)):
                raise ValueError(
                    "D-SigLIP requires binary positives; use false-negative masking"
                )
            equivalent = weights.bool()
            if not torch.all(equivalent.diagonal()):
                raise ValueError("Every EEG occurrence requires its diagonal target")
        if candidate_mask is None:
            valid = torch.ones_like(scores, dtype=torch.bool)
        else:
            valid = candidate_mask.to(device=scores.device, dtype=torch.bool).clone()
            if valid.shape != scores.shape:
                raise ValueError("candidate_mask must have the score-matrix shape")
        if not torch.all(valid.diagonal()):
            raise ValueError("A diagonal D-SigLIP positive was masked")

        # D-SigLIP: repeated equivalent candidates are neither negatives nor
        # additional positives.  Each anchor retains only its own diagonal.
        valid &= ~equivalent
        valid |= diagonal
        targets = diagonal.to(dtype=scores.dtype)
        pair_losses = F.binary_cross_entropy_with_logits(
            scores,
            targets,
            reduction="none",
        )
        return (pair_losses * valid.to(pair_losses.dtype)).sum() / count
