from data.pl_delay_sweep import common_delay_support
from data.pl_speech import PL_WINDOW_SCHEMA_VERSION, PLSpeechWindow


def _window(identifier: str, delay: float) -> PLSpeechWindow:
    return PLSpeechWindow(
        window_schema_version=PL_WINDOW_SCHEMA_VERSION,
        window_id=f"{identifier}-{delay}",
        record_id=identifier,
        block_id=f"block-{identifier}",
        split_group_id=f"group-{identifier}",
        split="train",
        subject_group_id="subject-1",
        subject_id="01",
        session_id="littleprince",
        run_id="11",
        speaker_id="f1",
        stimulus_position=1,
        char_count=10,
        eeg_file="fake.eeg",
        eeg_sampling_rate_hz=250,
        eeg_start_sample=round(delay / 1000 * 250),
        eeg_stop_sample=750 + round(delay / 1000 * 250),
        eeg_sample_count=750,
        valid_eeg_samples=750,
        padded_eeg_samples=0,
        audio_file="fake.wav",
        audio_source_sample_rate_hz=12_000,
        audio_start_sample=0,
        audio_stop_sample=36_000,
        audio_start_sec=0.0,
        audio_stop_sec=3.0,
        audio_target_id=f"target-{identifier}",
        window_offset_sec=0.0,
        window_sec=3.0,
        stride_sec=3.0,
        eeg_delay_ms=delay,
        source_trial_eeg_duration_sec=4.0,
        overlap_source="verified",
        quality_flag="ok",
    )


def test_delay_sweep_keeps_only_identical_support() -> None:
    first = [_window("a", 0.0), _window("b", 0.0)]
    second = [_window("a", 100.0)]
    common = common_delay_support({0.0: first, 100.0: second})
    assert [window.record_id for window in common[0.0]] == ["a"]
    assert [window.record_id for window in common[100.0]] == ["a"]
    assert common[100.0][0].eeg_start_sample > common[0.0][0].eeg_start_sample
