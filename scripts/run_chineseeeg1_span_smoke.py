"""Read real ChineseEEG1 spans and verify fixed tensor shapes without padding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.chineseeeg1_span_dataset import (
    ChineseEEG1SpanDataset,
    OfficialBrainVisionSegmentReader,
    collate_fixed_character_spans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--span-index",
        default="metadata/generated/chineseeeg1_character_spans_seed42.parquet",
    )
    parser.add_argument(
        "--output",
        default="reports/chineseeeg1_character_span_smoke.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for span_length in (4, 6, 8):
        reader = OfficialBrainVisionSegmentReader()
        dataset = ChineseEEG1SpanDataset(
            args.span_index,
            partition="test",
            span_char_count=span_length,
            eeg_reader=reader,
        )
        batch = collate_fixed_character_spans([dataset[0], dataset[1]])
        eeg = batch["eeg"]
        records.append(
            {
                "span_char_count": span_length,
                "test_span_count": len(dataset),
                "batch_shape": list(eeg.shape),
                "finite": bool(torch.isfinite(eeg).all()),
                "mean_absolute_value": float(eeg.abs().mean()),
                "fallback_header_count": reader.fallback_read_count,
                "padding_mask_present": "padding_mask" in batch,
            }
        )
    report = {
        "span_index": Path(args.span_index).as_posix(),
        "records": records,
        "all_finite": all(record["finite"] for record in records),
        "any_padding_mask": any(record["padding_mask_present"] for record in records),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
