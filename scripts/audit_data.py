"""Run the complete read-only EEG/text/audio audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.audit import audit_to_markdown, build_full_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chineseeeg1-root",
        type=Path,
        default=Path("D:/dataset/ChineseEEG/derivatives/preproc/filtered_0.5_30"),
    )
    parser.add_argument(
        "--chineseeeg1-derivatives-root",
        type=Path,
        default=Path("D:/dataset/ChineseEEG/derivatives"),
    )
    parser.add_argument(
        "--pl-root",
        type=Path,
        default=Path(
            "D:/dataset/ChineseEEG-2/PassiveListening/derivatives/preprocessed"
        ),
    )
    parser.add_argument(
        "--ra-root",
        type=Path,
        default=Path(
            "D:/dataset/ChineseEEG-2/ReadingAloud/derivatives/preprocessed"
        ),
    )
    parser.add_argument(
        "--materials-root",
        type=Path,
        default=Path("D:/dataset/ChineseEEG-2/materials&embeddings"),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    audit = build_full_audit(
        chineseeeg1_root=args.chineseeeg1_root,
        chineseeeg2_pl_root=args.pl_root,
        chineseeeg2_ra_root=args.ra_root,
        chineseeeg1_derivatives_root=args.chineseeeg1_derivatives_root,
        chineseeeg2_materials_root=args.materials_root,
    )
    json_text = json.dumps(audit, ensure_ascii=False, indent=2)
    markdown_text = audit_to_markdown(audit)
    if args.json_output:
        _write(args.json_output, json_text + "\n")
    if args.markdown_output:
        _write(args.markdown_output, markdown_text + "\n")
    if not args.json_output and not args.markdown_output:
        print(json_text)


if __name__ == "__main__":
    main()
