"""Common-support construction for fair EEG-delay comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from data.pl_speech import PLSpeechWindow


def delay_support_key(window: PLSpeechWindow) -> tuple[str, str, float, float]:
    """Identity of an audio/query unit before shifting its EEG span."""

    return (
        window.record_id,
        window.audio_target_id,
        window.window_offset_sec,
        window.window_sec,
    )


def common_delay_support(
    windows_by_delay: Mapping[float, Sequence[PLSpeechWindow]],
) -> dict[float, list[PLSpeechWindow]]:
    """Keep exactly the same underlying trial/audio units for every delay."""

    if len(windows_by_delay) < 2:
        raise ValueError("Delay sweep requires at least two delay settings")
    indexed: dict[float, dict[tuple[str, str, float, float], PLSpeechWindow]] = {}
    for delay, windows in windows_by_delay.items():
        by_key: dict[tuple[str, str, float, float], PLSpeechWindow] = {}
        for window in windows:
            if window.eeg_delay_ms != delay:
                raise ValueError(
                    f"Window delay {window.eeg_delay_ms} does not match key {delay}"
                )
            key = delay_support_key(window)
            if key in by_key:
                raise ValueError(f"Duplicate support key at delay={delay}: {key}")
            by_key[key] = window
        indexed[delay] = by_key
    shared_keys = set.intersection(*(set(values) for values in indexed.values()))
    if not shared_keys:
        raise ValueError("Delay settings have no common eligible support")

    reference_delay = sorted(indexed)[0]
    reference = indexed[reference_delay]
    for delay, by_key in indexed.items():
        for key in shared_keys:
            left = reference[key]
            right = by_key[key]
            signature_left = (
                left.split,
                left.split_group_id,
                left.audio_start_sample,
                left.audio_stop_sample,
                left.audio_target_id,
                left.subject_group_id,
            )
            signature_right = (
                right.split,
                right.split_group_id,
                right.audio_start_sample,
                right.audio_stop_sample,
                right.audio_target_id,
                right.subject_group_id,
            )
            if signature_left != signature_right:
                raise ValueError(
                    f"Non-delay metadata changed at delay={delay}: {key}"
                )
    return {
        delay: sorted(
            (indexed[delay][key] for key in shared_keys),
            key=lambda window: window.window_id,
        )
        for delay in sorted(indexed)
    }
