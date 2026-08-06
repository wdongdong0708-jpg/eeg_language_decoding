import torch
from torch import nn

from models.losses import MaskedSoftTargetContrastiveLoss
from models.projection_head import AttentionPool1d, PooledProjectionHead
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
