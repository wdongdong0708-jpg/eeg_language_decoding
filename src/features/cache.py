"""Stable feature-cache keys that include model and block provenance."""

from __future__ import annotations

import hashlib
import json
import re

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


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


def safe_artifact_filename(identifier: str, *, suffix: str = ".npz") -> str:
    """Create a readable Windows-safe name without trusting manifest identifiers."""

    if not identifier:
        raise ValueError("Artifact identifier cannot be empty")
    readable = _SAFE_FILENAME.sub("_", identifier).strip("._")[:80] or "artifact"
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{readable}-{digest}{normalized_suffix}"
