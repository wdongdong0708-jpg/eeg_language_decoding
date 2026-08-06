"""Text/audio feature extraction and provenance-aware caches."""

from features.audio_features import (
    DEFAULT_AUDIO_LAYERS,
    AudioFeatureConfig,
    AudioFeatureInput,
    AudioFrameFeatureResult,
    Wav2VecFrameExtractor,
    average_hidden_layers,
    interpolate_audio_sequence,
    load_audio_frame_features,
    load_audio_sequence_features,
    mean_pool_audio_frames,
    save_audio_frame_features,
    save_audio_sequence_features,
)
from features.cache import feature_cache_key, safe_artifact_filename
from features.model_loading import (
    enable_strict_huggingface_offline_mode,
    resolve_model_source,
)
from features.text_features import (
    STATIC_CHARACTER_BASELINE_LAYER_INDEX,
    TextEmbeddingExtractor,
    TextFeatureConfig,
    TextFeatureInput,
    TextFeatureResult,
    load_text_features,
    pool_text_span,
    save_text_features,
    select_text_span_by_offsets,
)

__all__ = [
    "AudioFeatureConfig",
    "AudioFeatureInput",
    "AudioFrameFeatureResult",
    "DEFAULT_AUDIO_LAYERS",
    "TextEmbeddingExtractor",
    "TextFeatureConfig",
    "TextFeatureInput",
    "TextFeatureResult",
    "STATIC_CHARACTER_BASELINE_LAYER_INDEX",
    "Wav2VecFrameExtractor",
    "average_hidden_layers",
    "feature_cache_key",
    "enable_strict_huggingface_offline_mode",
    "load_audio_frame_features",
    "load_audio_sequence_features",
    "load_text_features",
    "pool_text_span",
    "mean_pool_audio_frames",
    "interpolate_audio_sequence",
    "resolve_model_source",
    "safe_artifact_filename",
    "save_audio_frame_features",
    "save_audio_sequence_features",
    "save_text_features",
    "select_text_span_by_offsets",
]
