import numpy as np

from preprocessing.eeg import ChannelRobustScaler, fit_channel_robust_scaler


def test_robust_scaler_fits_union_of_overlapping_train_intervals() -> None:
    source = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
            [10.0, 11.0, 12.0, 13.0, 14.0, -100.0],
        ],
        dtype=np.float32,
    )
    rows = [
        {
            "split": "train",
            "eeg_file": "record.eeg",
            "eeg_start_sample": 0,
            "eeg_stop_sample": 4,
        },
        {
            "split": "train",
            "eeg_file": "record.eeg",
            "eeg_start_sample": 2,
            "eeg_stop_sample": 6,
        },
    ]

    def reader(_: str, start: int, stop: int) -> np.ndarray:
        return source[:, start:stop]

    scaler = fit_channel_robust_scaler(rows, eeg_reader=reader, clamp=2.0)
    assert scaler.fitted_span_count == 2
    assert scaler.fitted_unique_sample_count == 6
    low, median, high = np.quantile(source, (0.25, 0.5, 0.75), axis=1)
    np.testing.assert_allclose(scaler.centers["record.eeg"], median)
    np.testing.assert_allclose(scaler.scales["record.eeg"], high - low)
    transformed = scaler.transform("record.eeg", source)
    assert transformed.min() >= -2.0
    assert transformed.max() <= 2.0
    restored = ChannelRobustScaler.from_state_dict(scaler.state_dict())
    np.testing.assert_allclose(
        restored.transform("record.eeg", source), transformed
    )
