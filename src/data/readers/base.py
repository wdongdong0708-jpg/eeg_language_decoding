"""Shared reader interfaces and BIDS discovery helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from data.manifest import ManifestRecord


class DatasetReader(Protocol):
    def discover(self) -> list[Path]: ...

    def read_blocks(self) -> list[ManifestRecord]: ...


def discover_brainvision_headers(root: str | Path) -> list[Path]:
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"EEG root does not exist: {path}")
    return sorted(path.rglob("*_eeg.vhdr"))

