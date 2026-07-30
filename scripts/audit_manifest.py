"""Audit the generated trial manifest and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.manifest_audit import audit_manifest, render_manifest_audit_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="metadata/all_trials.parquet")
    parser.add_argument("--csv", default="metadata/all_trials.csv")
    parser.add_argument("--data-audit", default="reports/data_audit.json")
    parser.add_argument(
        "--build-diagnostics",
        default="metadata/manifest_build_diagnostics.json",
    )
    parser.add_argument("--json", default="reports/manifest_audit.json")
    parser.add_argument("--markdown", default="reports/manifest_audit.md")
    args = parser.parse_args()

    result = audit_manifest(
        parquet_path=args.parquet,
        csv_path=args.csv,
        data_audit_path=args.data_audit,
        build_diagnostics_path=args.build_diagnostics,
    )
    json_path = Path(args.json)
    markdown_path = Path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_manifest_audit_markdown(result),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "row_count": result["manifest"]["row_count"],
                "quality_flags": result["quality_flags"],
                "splits": result["splits"],
                "parquet_csv_consistency": result["parquet_csv_consistency"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
