import torch
import torch.nn.functional as F

from models.losses import DeduplicatedSigLipLoss


def test_d_siglip_masks_duplicate_off_diagonal_targets() -> None:
    estimates = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    candidates = estimates.clone()
    positives = torch.tensor(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    objective = DeduplicatedSigLipLoss(
        logit_scale_init=10.0,
        bias_init=-10.0,
        learn_temperature=False,
        learn_bias=False,
    )
    scores = objective.get_scores(estimates, candidates)
    valid = torch.tensor(
        [[True, False, True], [False, True, True], [True, True, True]]
    )
    targets = torch.eye(3)
    expected = (
        F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        * valid
    ).sum() / 3
    observed = objective(
        estimates,
        candidates,
        positive_weights=positives,
        candidate_mask=torch.ones(3, 3, dtype=torch.bool),
    )
    assert torch.allclose(observed, expected)


def test_d_siglip_learns_scale_and_bias() -> None:
    objective = DeduplicatedSigLipLoss()
    estimates = torch.randn(4, 8, requires_grad=True)
    candidates = torch.randn(4, 8)
    loss = objective(estimates, candidates)
    loss.backward()
    assert objective.logit_scale.grad is not None
    assert objective.bias.grad is not None
    assert estimates.grad is not None
