"""Stable feature-cache keys that include model and block provenance."""

from __future__ import annotations

import hashlib
import json


def feature_cache_key(
    *,
    modality: str,
    source_fingerprint: str,
    block_id: str,
    model_id: str,
    pooling: str,
    extractor_version: str,
) -> str:
    payload = {
        "modality": modality,
        "source_fingerprint": source_fingerprint,
        "block_id": block_id,
        "model_id": model_id,
        "pooling": pooling,
        "extractor_version": extractor_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "feature-" + hashlib.sha256(canonical.encode()).hexdigest()[:24]

