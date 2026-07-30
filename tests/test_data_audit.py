from data.audit import pair_row_trials


def test_row_trial_pairing_drops_orphan_end_without_fabricating_start() -> None:
    events = [
        {"sample": 1, "trial_type": "ROWE"},
        {"sample": 10, "trial_type": "ROWS"},
        {"sample": 20, "trial_type": "ROWE"},
    ]
    pairs, orphan_starts, orphan_ends = pair_row_trials(events)
    assert pairs == [(10, 20)]
    assert orphan_starts == 0
    assert orphan_ends == 1


def test_nested_start_is_reported() -> None:
    events = [
        {"sample": 10, "trial_type": "ROWS"},
        {"sample": 12, "trial_type": "ROWS"},
        {"sample": 20, "trial_type": "ROWE"},
    ]
    pairs, orphan_starts, orphan_ends = pair_row_trials(events)
    assert pairs == [(12, 20)]
    assert orphan_starts == 1
    assert orphan_ends == 0

