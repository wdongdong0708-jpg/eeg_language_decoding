"""Guardrails for consuming official EEG derivatives without hidden reprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EEGPreprocessingPolicy:
    apply_filter: bool = False
    apply_notch: bool = False
    apply_ica: bool = False
    interpolate_bad_channels: bool = False
    rereference: bool = False
    resample: bool = False

    def validate_for_official_derivative(self) -> None:
        enabled = [
            field
            for field in (
                "apply_filter",
                "apply_notch",
                "apply_ica",
                "interpolate_bad_channels",
                "rereference",
                "resample",
            )
            if getattr(self, field)
        ]
        if enabled:
            raise ValueError(
                "Official derivatives cannot be reprocessed by default; "
                f"enabled operations: {enabled}"
            )

