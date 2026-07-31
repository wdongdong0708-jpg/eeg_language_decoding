import pytest
import torch

from models.eeg_encoder import (
    DilatedConvBlock,
    DilatedSimpleConv,
    MetaAlignedConvNoSubject,
    build_eeg_encoder,
)
from models.losses import ClipContrastiveLoss, sequence_similarity
from models.retrieval_model import EEGSpeechRetrievalModel


def test_dilated_encoder_preserves_time_and_projects_channels() -> None:
    encoder = DilatedSimpleConv(
        input_channels=8,
        output_channels=16,
        hidden_channels=12,
        depth=4,
        kernel_size=5,
        dilation_growth=2,
    )
    output = encoder(torch.randn(3, 8, 75))
    assert output.shape == (3, 16, 75)


def test_even_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        DilatedConvBlock(4, 4, kernel_size=4, dilation=1)


def test_meta_aligned_encoder_has_periodic_dilation_and_preserves_time() -> None:
    encoder = MetaAlignedConvNoSubject(
        input_channels=8,
        output_channels=16,
        initial_channels=10,
        hidden_channels=12,
        depth=6,
        kernel_size=3,
        dilation_period=3,
        glu_every=2,
    )
    output = encoder(torch.randn(3, 8, 75))
    assert output.shape == (3, 16, 75)
    assert encoder.dilations == [1, 2, 4, 1, 2, 4]
    assert isinstance(encoder.glus[1][-1], torch.nn.GLU)
    assert isinstance(encoder.glus[0], torch.nn.Identity)


def test_encoder_factory_builds_meta_no_subject_topology() -> None:
    encoder = build_eeg_encoder(
        {
            "name": "meta_aligned_conv_no_subject",
            "initial_channels": 10,
            "hidden_channels": 12,
            "depth": 2,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 5,
            "glu_every": 2,
            "glu_context": 1,
            "batch_norm": True,
            "skip": True,
        },
        input_channels=8,
        output_channels=16,
    )
    assert isinstance(encoder, MetaAlignedConvNoSubject)


def test_sequence_similarity_normalization_matches_manual_result() -> None:
    estimates = torch.tensor([[[3.0, 4.0]]])
    candidates = torch.tensor([[[6.0, 8.0]], [[0.0, 2.0]]])
    scores = sequence_similarity(estimates, candidates, norm_kind="y")
    assert scores == pytest.approx(torch.tensor([[5.0, 4.0]]))


def test_clip_loss_prefers_aligned_sequences_and_backpropagates() -> None:
    targets = torch.eye(4).reshape(4, 1, 4)
    estimates = targets.clone().requires_grad_(True)
    objective = ClipContrastiveLoss(norm_kind="xy", temperature=0.07)
    aligned_loss = objective(estimates, targets)
    shuffled_loss = objective(estimates, targets.roll(1, dims=0))
    assert aligned_loss < shuffled_loss
    aligned_loss.backward()
    assert estimates.grad is not None
    assert torch.isfinite(estimates.grad).all()


def test_retrieval_model_connects_encoder_and_loss() -> None:
    model = EEGSpeechRetrievalModel(
        DilatedSimpleConv(4, 6, hidden_channels=8, depth=2),
        ClipContrastiveLoss(norm_kind="xy"),
    )
    eeg = torch.randn(2, 4, 20)
    target = torch.randn(2, 6, 20)
    loss = model.compute_loss(eeg, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
