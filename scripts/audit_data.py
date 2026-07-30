"""Audit dataset paths and count BIDS BrainVision/event files.

This first-stage audit is intentionally read-only. Deeper row/text/audio
alignment checks will be added after the manifest schema is reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ROOTS = {
    "chineseeeg1": Path(
        "D:/dataset/ChineseEEG/derivatives/preproc/filtered_0.5_30"
    ),
    "chineseeeg2_pl": Path(
        "D:/dataset/ChineseEEG-2/PassiveListening/derivatives/preprocessed"
    ),
    "chineseeeg2_ra": Path(
        "D:/dataset/ChineseEEG-2/ReadingAloud/derivatives/preprocessed"
    ),
}


def audit_root(root: Path) -> dict[str, object]:
    if not root.is_dir():
        return {"root": str(root), "exists": False}
    headers = sorted(root.rglob("*_eeg.vhdr"))
    events = sorted(root.rglob("*_events.tsv"))
    return {
        "root": str(root.resolve()),
        "exists": True,
        "brainvision_headers": len(headers),
        "event_tables": len(events),
        "header_event_count_match": len(headers) == len(events),
        "subjects": sorted(path.name for path in root.glob("sub-*") if path.is_dir()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Optional dataset root; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.root if args.root else list(DEFAULT_ROOTS.values())
    print(json.dumps([audit_root(root) for root in roots], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

