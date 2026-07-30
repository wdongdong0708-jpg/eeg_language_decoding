import pytest

from preprocessing.eeg import EEGPreprocessingPolicy


def test_default_policy_preserves_official_derivative() -> None:
    EEGPreprocessingPolicy().validate_for_official_derivative()


def test_hidden_reprocessing_is_rejected() -> None:
    with pytest.raises(ValueError, match="apply_ica"):
        EEGPreprocessingPolicy(apply_ica=True).validate_for_official_derivative()

