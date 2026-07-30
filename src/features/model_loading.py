"""Deterministic pretrained-model source resolution."""

from __future__ import annotations

import os
from pathlib import Path

STRICT_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def enable_strict_huggingface_offline_mode() -> None:
    """Disable Hub HTTP access for the remainder of the current process."""

    os.environ.update(STRICT_OFFLINE_ENVIRONMENT)


def resolve_model_source(
    model_id_or_path: str,
    *,
    local_files_only: bool,
) -> str:
    """Resolve a cached repo ID to a directory before Transformers sees it.

    Some Transformers/PEFT combinations probe ``adapter_config.json`` over HTTP
    even when ``local_files_only=True`` was passed to ``AutoModel``. Resolving
    the cached snapshot first and passing its directory avoids that remote
    adapter-probe branch. No network fallback is allowed in strict offline mode.
    """

    requested = str(model_id_or_path).strip()
    if not requested:
        raise ValueError("model_id_or_path cannot be empty")
    if not local_files_only:
        return requested

    enable_strict_huggingface_offline_mode()
    local_path = Path(requested).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            repo_id=requested,
            local_files_only=True,
        )
    except Exception as error:
        raise FileNotFoundError(
            f"No complete cached Hugging Face snapshot is available for "
            f"{requested!r}; strict offline mode forbids a network fallback"
        ) from error

    resolved_path = Path(resolved)
    if not resolved_path.is_dir():
        raise FileNotFoundError(
            f"Cached Hugging Face snapshot is not a directory: {resolved_path}"
        )
    return str(resolved_path.resolve())
