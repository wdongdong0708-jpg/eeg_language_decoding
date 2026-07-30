"""Quality-control report contracts for events, channels and alignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class QualityFinding:
    check_name: str
    severity: Severity
    source_id: str
    message: str

