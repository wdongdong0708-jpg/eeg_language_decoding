"""Audit whether ChineseEEG1 supports an approximate visual character clock."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.chineseeeg1_timeline import (
    OFFICIAL_PRESENTATION_FILE,
    audit_chineseeeg1_timeline,
    inspect_official_presentation_code,
    write_timeline_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="metadata/all_trials.parquet")
    parser.add_argument(
        "--official-code-root",
        required=True,
        help=(
            "Existing local checkout of ncclabsustech/"
            "Chinese_reading_task_eeg_processing. The script never downloads it."
        ),
    )
    parser.add_argument(
        "--json-output",
        default="reports/chineseeeg1_character_timeline_audit.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/chineseeeg1_character_timeline_audit.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    source = Path(args.official_code_root) / OFFICIAL_PRESENTATION_FILE
    evidence = inspect_official_presentation_code(source)
    frame = pd.read_parquet(manifest)
    report = audit_chineseeeg1_timeline(
        frame,
        code_evidence=evidence,
        manifest_path=manifest,
    )
    write_timeline_audit(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    print(f"verdict={report['verdict']}")
    print(f"configured_pace_verified={report['configured_pace_verified']}")
    print(
        "exact_character_onsets_verified="
        f"{report['exact_character_onsets_verified']}"
    )


if __name__ == "__main__":
    main()
