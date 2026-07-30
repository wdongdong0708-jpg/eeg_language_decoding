"""Audio feature definitions."""

OFFICIAL_AUDIO_MODEL = "airesearch/wav2vec2-large-xlsr-53-th"
OFFICIAL_AUDIO_POOLING = "last_hidden_state.mean(dim=1)"
MODEL_SAMPLE_RATE_HZ = 16_000

