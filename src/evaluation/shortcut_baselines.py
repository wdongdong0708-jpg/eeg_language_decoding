"""Required shortcut baselines and report validation."""

REQUIRED_SHORTCUT_BASELINES = frozenset(
    {
        "random",
        "duration_only",
        "character_count_only",
        "padding_mask_only",
        "sentence_position_only",
        "subject_id_only",
        "audio_envelope",
    }
)


def validate_shortcut_baseline_names(names: set[str]) -> None:
    missing = REQUIRED_SHORTCUT_BASELINES - names
    if missing:
        raise ValueError(f"Missing required shortcut baselines: {sorted(missing)}")

