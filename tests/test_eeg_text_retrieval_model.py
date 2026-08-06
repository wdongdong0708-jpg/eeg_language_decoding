import torch
from torch import nn

from models.losses import MaskedSoftTargetContrastiveLoss
from models.projection_head import (
    AttentionPool1d,
    BahdanauAttention,
    PooledProjectionHead,
)
from models.retrieval_model import EEGTextRetrievalModel


def test_eeg_text_model_pools_fixed_eeg_and_character_states() -> None:
    model = EEGTextRetrievalModel(
        eeg_encoder=nn.Conv1d(3, 6, 3, padding=1),
        eeg_projection=PooledProjectionHead(
            6,
            4,
            sequence_axis="last",
            pooling="attention",
        ),
        text_sequence_pooler=AttentionPool1d(8),
        text_projection=nn.Linear(8, 4),
        objective=MaskedSoftTargetContrastiveLoss(symmetric=True),
    )
    eeg = torch.randn(3, 3, 358)
    text = torch.randn(3, 4, 8)
    policy = torch.eye(3)
    loss = model.compute_loss(
        eeg,
        text,
        positive_weights=policy,
        candidate_mask=torch.ones(3, 3, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.eeg_projection.attention.query.grad is not None
    assert model.text_sequence_pooler.query.grad is not None


def test_eeg_text_model_can_emit_normalized_static_embeddings() -> None:
    model = EEGTextRetrievalModel(
        eeg_encoder=nn.Conv1d(3, 4, 3, padding=1),
        eeg_projection=PooledProjectionHead(
            4, 4, sequence_axis="last", pooling="attention"
        ),
        text_projection=nn.Identity(),
        objective=MaskedSoftTargetContrastiveLoss(),
        normalize_embeddings=True,
    )
    eeg = model.encode_eeg(torch.randn(3, 3, 20))
    text = model.encode_text(torch.randn(3, 4))
    assert torch.allclose(eeg.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(text.norm(dim=1), torch.ones(3), atol=1e-5)



def test_bahdanau_attention_pools_feature_first_sequence() -> None:
    pooling = BahdanauAttention(input_size=5, hidden_size=11)
    keys = torch.randn(3, 5, 17, requires_grad=True)

    weights = pooling.attention_weights(keys)
    output = pooling(keys)

    assert weights.shape == (3, 1, 17)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 1), atol=1e-6)
    assert output.shape == (3, 5, 1)
    output.square().sum().backward()
    assert pooling.Wa.weight.grad is not None
    assert pooling.Va.weight.grad is not None


def test_bahdanau_attention_supports_optional_queries() -> None:
    pooling = BahdanauAttention(input_size=5, hidden_size=11)
    keys = torch.randn(3, 5, 17)
    queries = torch.randn(3, 5, 17, requires_grad=True)

    pooling(keys, queries).sum().backward()

    assert pooling.Ua.weight.grad is not None


def test_pooled_projection_head_supports_bahdanau_attention() -> None:
    projection = PooledProjectionHead(
        5,
        7,
        sequence_axis="last",
        pooling="bahdanau_attention",
        bahdanau_attention_hidden_size=11,
    )

    output = projection(torch.randn(3, 5, 17))

    assert output.shape == (3, 7)
