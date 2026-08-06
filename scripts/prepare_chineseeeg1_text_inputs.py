"""Export unique Stage-2 local or full-sentence texts for frozen encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--span-index",
        default="metadata/generated/chineseeeg1_character_spans_seed42.parquet",
    )
    parser.add_argument("--scope", choices=["local", "sentence"], required=True)
    parser.add_argument("--span-length", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="Deterministic prefix for smoke tests; omit for the complete input set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scope == "local":
        identifier, text_column = "span_text_id", "span_text"
    else:
        identifier, text_column = "global_text_id", "source_sentence_text"
    filters = (
        [("span_char_count", "=", args.span_length)]
        if args.span_length is not None
        else None
    )
    table = pq.read_table(
        args.span_index,
        columns=[identifier, text_column],
        filters=filters,
    )
    grouped = table.group_by(identifier).aggregate([(text_column, "count_distinct")])
    count_column = f"{text_column}_count_distinct"
    if pc.any(pc.not_equal(grouped[count_column], 1)).as_py():
        raise ValueError(f"One {identifier} maps to multiple texts")
    unique = table.group_by(identifier).aggregate([(text_column, "min")])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(unique.to_pylist(), key=lambda item: str(item[identifier]))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        ordered_rows = ordered_rows[: args.limit]
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered_rows:
            handle.write(
                json.dumps(
                    {
                        "content_id": str(row[identifier]),
                        "text": str(row[f"{text_column}_min"]),
                        "encoding_scope": args.scope,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"texts={len(ordered_rows)}")
    print(f"output={target}")


if __name__ == "__main__":
    main()
