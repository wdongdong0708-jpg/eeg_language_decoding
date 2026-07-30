"""Build the deterministic complete stimulus-row trial manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.trial_manifest import (
    DEFAULT_SPLIT_SEED,
    ManifestPaths,
    build_trial_manifest,
    default_manifest_paths,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", default="reports/data_audit.json")
    parser.add_argument("--parquet", default="metadata/all_trials.parquet")
    parser.add_argument("--csv", default="metadata/all_trials.csv")
    parser.add_argument(
        "--normalization-rules",
        default="metadata/normalization_rules.json",
    )
    parser.add_argument(
        "--alignment-overrides",
        default="metadata/text_alignment_overrides.csv",
    )
    parser.add_argument(
        "--diagnostics",
        default="metadata/manifest_build_diagnostics.json",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--chineseeeg1-eeg-root")
    parser.add_argument("--chineseeeg1-novel-root")
    parser.add_argument("--chineseeeg2-pl-root")
    parser.add_argument("--chineseeeg2-ra-root")
    parser.add_argument("--chineseeeg2-material-root")
    parser.add_argument("--chineseeeg2-audio-root")
    return parser


def main() -> None:
    args = _parser().parse_args()
    defaults = default_manifest_paths(audit_json=args.audit_json)
    paths = ManifestPaths(
        chineseeeg1_eeg_root=Path(
            args.chineseeeg1_eeg_root or defaults.chineseeeg1_eeg_root
        ),
        chineseeeg1_novel_root=Path(
            args.chineseeeg1_novel_root or defaults.chineseeeg1_novel_root
        ),
        chineseeeg2_pl_root=Path(
            args.chineseeeg2_pl_root or defaults.chineseeeg2_pl_root
        ),
        chineseeeg2_ra_root=Path(
            args.chineseeeg2_ra_root or defaults.chineseeeg2_ra_root
        ),
        chineseeeg2_material_root=Path(
            args.chineseeeg2_material_root or defaults.chineseeeg2_material_root
        ),
        chineseeeg2_audio_root=Path(
            args.chineseeeg2_audio_root or defaults.chineseeeg2_audio_root
        ),
        audit_json=Path(args.audit_json),
    )
    for name, path in (
        ("ChineseEEG1 EEG", paths.chineseeeg1_eeg_root),
        ("ChineseEEG1 novels", paths.chineseeeg1_novel_root),
        ("ChineseEEG2 PL", paths.chineseeeg2_pl_root),
        ("ChineseEEG2 RA", paths.chineseeeg2_ra_root),
        ("ChineseEEG2 materials", paths.chineseeeg2_material_root),
        ("ChineseEEG2 audio", paths.chineseeeg2_audio_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} root does not exist: {path}")

    result = build_trial_manifest(
        paths=paths,
        parquet_path=args.parquet,
        csv_path=args.csv,
        normalization_rules_path=args.normalization_rules,
        override_path=args.alignment_overrides,
        diagnostics_path=args.diagnostics,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
