"""Explicit dataset-path resolution with optional environment overrides."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_existing_path(default: str | Path, *, env_var: str | None = None) -> Path:
    raw = os.environ.get(env_var) if env_var else None
    path = Path(raw if raw else default)
    if not path.exists():
        source = f"${env_var}" if raw and env_var else "configured default"
        raise FileNotFoundError(f"{source} path does not exist: {path}")
    return path.resolve()

