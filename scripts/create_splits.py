"""Build independent, versioned protocol-level split artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.protocol_audit import build_protocol_audit, render_protocol_audit_markdown
from data.protocol_splitting import (
    DEFAULT_PROTOCOL_SEED,
    build_all_protocols,
    load_protocol_manifest,
    write_json_deterministic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="metadata/all_trials.parquet")
    parser.add_argument("--seed", type=int, default=DEFAULT_PROTOCOL_SEED)
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--audit-json", default="reports/split_protocol_audit.json")
    parser.add_argument("--audit-markdown", default="reports/split_protocol_audit.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_protocol_manifest(args.manifest)
    artifacts = build_all_protocols(
        rows,
        manifest_path=Path(args.manifest).as_posix(),
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    hashes = {
        name: write_json_deterministic(output_dir / name, artifact)
        for name, artifact in artifacts.items()
    }
    audit = build_protocol_audit(artifacts, artifact_sha256=hashes)
    write_json_deterministic(args.audit_json, audit)
    markdown_path = Path(args.audit_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_protocol_audit_markdown(audit),
        encoding="utf-8",
        newline="\n",
    )
    for name in sorted(hashes):
        print(f"{name}: {hashes[name]}")


if __name__ == "__main__":
    main()
