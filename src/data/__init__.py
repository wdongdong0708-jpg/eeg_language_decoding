"""Data contracts, readers, deterministic splitting and alignment utilities."""

from data.manifest import ManifestRecord, validate_manifest_records
from data.splitting import SplitRatios, assign_split
from data.text_normalization import make_content_id, normalize_text

__all__ = [
    "ManifestRecord",
    "SplitRatios",
    "assign_split",
    "make_content_id",
    "normalize_text",
    "validate_manifest_records",
]

