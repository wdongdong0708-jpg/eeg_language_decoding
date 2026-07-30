"""Generate windows that remain wholly attributable to one verified block."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

TailPolicy = Literal["drop", "pad"]


@dataclass(frozen=True, slots=True)
class WindowSpec:
    size_samples: int
    stride_samples: int
    tail_policy: TailPolicy = "drop"

    def validate(self) -> None:
        if self.size_samples <= 0 or self.stride_samples <= 0:
            raise ValueError("Window size and stride must be positive")
        if self.tail_policy not in {"drop", "pad"}:
            raise ValueError(f"Unknown tail policy: {self.tail_policy}")


@dataclass(frozen=True, slots=True)
class BlockWindow:
    window_id: str
    block_id: str
    content_id: str
    split: str
    start_sample: int
    stop_sample: int
    valid_samples: int
    padded_samples: int


def _window_id(block_id: str, start_sample: int, stop_sample: int) -> str:
    payload = f"{block_id}\0{start_sample}\0{stop_sample}".encode()
    return "window-" + hashlib.sha256(payload).hexdigest()[:20]


def make_block_windows(
    *,
    block_id: str,
    content_id: str,
    split: str,
    block_start_sample: int,
    block_stop_sample: int,
    spec: WindowSpec,
) -> list[BlockWindow]:
    """Create windows after split assignment; no window can cross a block edge."""

    spec.validate()
    if not block_id or not content_id or not split:
        raise ValueError("block_id, content_id and inherited split are required")
    if block_start_sample < 0 or block_stop_sample <= block_start_sample:
        raise ValueError("Block sample boundaries must be positive and ordered")

    windows: list[BlockWindow] = []
    start = block_start_sample
    while start < block_stop_sample:
        requested_stop = start + spec.size_samples
        valid_stop = min(requested_stop, block_stop_sample)
        valid_samples = valid_stop - start
        if valid_samples < spec.size_samples and spec.tail_policy == "drop":
            break
        padded_samples = spec.size_samples - valid_samples
        windows.append(
            BlockWindow(
                window_id=_window_id(block_id, start, valid_stop),
                block_id=block_id,
                content_id=content_id,
                split=split,
                start_sample=start,
                stop_sample=valid_stop,
                valid_samples=valid_samples,
                padded_samples=padded_samples,
            )
        )
        if padded_samples:
            break
        start += spec.stride_samples

    return windows


def assert_windows_within_block(
    windows: list[BlockWindow],
    *,
    block_start_sample: int,
    block_stop_sample: int,
    expected_split: str,
) -> None:
    for window in windows:
        if window.start_sample < block_start_sample or window.stop_sample > block_stop_sample:
            raise ValueError(f"Window crosses its block boundary: {window.window_id}")
        if window.split != expected_split:
            raise ValueError(f"Window changed split: {window.window_id}")

