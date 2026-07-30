"""Dataset-specific readers.

Readers will emit validated :class:`data.manifest.ManifestRecord` objects and
must not decide sample-level splits.
"""

from data.readers.stimulus_text import (
    StimulusTextUnit,
    load_chineseeeg2_workbook,
    parse_chineseeeg2_rows,
)

__all__ = [
    "StimulusTextUnit",
    "load_chineseeeg2_workbook",
    "parse_chineseeeg2_rows",
]
