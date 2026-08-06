"""Shared factory for Stage-2 EEG-text training and checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from models.eeg_encoder import build_eeg_encoder
from models.losses import DeduplicatedSigLipLoss, MaskedSoftTargetContrastiveLoss
from models.projection_head import AttentionPool1d, PooledProjectionHead
from models.retrieval_model import EEGTextRetrievalModel


def build_eeg_text_retrieval_model(
    config: Mapping[str, object],
    *,
    eeg_channels: int,
    text_shape: Sequence[int],
    n_subjects: int | None = None,
) -> EEGTextRetrievalModel:
    """Build the exact pooled model used by both training and evaluation."""

    model_config = config["model"]
    training_config = config["training"]
    loss_config = config.get("loss", training_config)
    if not isinstance(model_config, Mapping) or not isinstance(
        training_config, Mapping
    ) or not isinstance(loss_config, Mapping):
        raise ValueError("Malformed model, loss, or training configuration")
    if not text_shape:
        raise ValueError("text_shape cannot be empty")
    text_dim = int(text_shape[-1])
    shared_dim = int(model_config["shared_dim"])
    encoder = build_eeg_encoder(
        dict(model_config["eeg_encoder"]),
        input_channels=eeg_channels,
        output_channels=shared_dim,
        n_subjects=n_subjects,
    )
    text_sequence_pooler: nn.Module | None = None
    if len(text_shape) == 2 and model_config["text_pooling"] == "attention":
        text_sequence_pooler = AttentionPool1d(text_dim)
    projection_kind = str(model_config.get("text_projection", "linear"))
    if projection_kind == "identity":
        if text_dim != shared_dim:
            raise ValueError("Identity text projection requires text_dim == shared_dim")
        text_projection: nn.Module = nn.Identity()
    elif projection_kind == "linear":
        text_projection = nn.Linear(text_dim, shared_dim)
    else:
        raise ValueError(f"Unknown text projection: {projection_kind}")

    loss_name = str(loss_config.get("name", "masked_soft_target_contrastive"))
    if loss_name == "d_siglip":
        objective: nn.Module = DeduplicatedSigLipLoss(
            logit_scale_init=float(loss_config.get("logit_scale_init", 10.0)),
            bias_init=float(loss_config.get("bias_init", -10.0)),
            learn_temperature=bool(loss_config.get("learn_temperature", True)),
            learn_bias=bool(loss_config.get("learn_bias", True)),
            normalize=bool(loss_config.get("normalize", True)),
        )
    elif loss_name == "masked_soft_target_contrastive":
        objective = MaskedSoftTargetContrastiveLoss(
            temperature=float(
                loss_config.get("temperature", training_config.get("temperature", 0.07))
            ),
            learn_temperature=bool(
                loss_config.get(
                    "learn_temperature",
                    training_config.get("learn_temperature", False),
                )
            ),
            symmetric=bool(
                loss_config.get(
                    "symmetric",
                    training_config.get("symmetric_loss", True),
                )
            ),
            normalize=bool(loss_config.get("normalize", True)),
        )
    else:
        raise ValueError(f"Unknown pooled retrieval loss: {loss_name}")

    return EEGTextRetrievalModel(
        eeg_encoder=encoder,
        eeg_projection=PooledProjectionHead(
            shared_dim,
            shared_dim,
            sequence_axis="last",
            pooling=str(model_config["eeg_temporal_pooling"]),
            dropout=float(model_config.get("projection_dropout", 0.1)),
        ),
        text_sequence_pooler=text_sequence_pooler,
        text_projection=text_projection,
        objective=objective,
        normalize_embeddings=bool(model_config.get("normalize_embeddings", False)),
    )
