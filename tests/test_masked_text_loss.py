import torch

from models.losses import MaskedSoftTargetContrastiveLoss


def test_masked_soft_target_loss_accepts_multi_positive_policy() -> None:
    estimates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.tensor([[1.0, 0.25], [0.0, 1.0]])
    mask = torch.ones((2, 2), dtype=torch.bool)
    objective = MaskedSoftTargetContrastiveLoss(symmetric=False)
    loss = objective(
        estimates,
        candidates,
        positive_weights=weights,
        candidate_mask=mask,
    )
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_loss_rejects_a_masked_positive() -> None:
    objective = MaskedSoftTargetContrastiveLoss(symmetric=False)
    estimates = torch.eye(2)
    weights = torch.eye(2)
    mask = torch.tensor([[False, True], [True, True]])
    try:
        objective(estimates, estimates, positive_weights=weights, candidate_mask=mask)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected a masked-positive error")
