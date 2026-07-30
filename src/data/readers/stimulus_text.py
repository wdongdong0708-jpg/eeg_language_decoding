"""Canonical stimulus-row parsing for ChineseEEG text workbooks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data.readers.simple_xlsx import XlsxRow, read_xlsx_sheet
from data.text_normalization import normalize_identifier, normalize_text

_CHAPTER_MARKER = re.compile(r"^[0-9]+$")


@dataclass(frozen=True, slots=True)
class StimulusTextUnit:
    dataset_id: str
    book_id: str
    segment_id: str
    excel_row: int
    sequence_index: int
    chapter_id: int
    row_in_chapter: int
    text: str
    normalized_text: str
    is_chapter_marker: bool
    f1_boundary_label: str | None = None
    m1_boundary_label: str | None = None


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_chineseeeg2_rows(
    rows: Iterable[XlsxRow],
    *,
    book_id: str,
    first_chapter_excel_row: int,
    dataset_id: str = "chineseeeg2",
) -> list[StimulusTextUnit]:
    """Parse row-level stimulus segments and explicit chapter-marker rows."""

    normalized_book = normalize_identifier(book_id)
    material_rows = list(rows)
    if not material_rows:
        raise ValueError("Stimulus workbook is empty")

    first_data = next((row for row in material_rows if row.excel_row == 2), None)
    if first_data is None:
        raise ValueError("Stimulus workbook is missing Excel row 2")
    first_chapter = next(
        (row for row in material_rows if row.excel_row == first_chapter_excel_row),
        None,
    )
    if first_chapter is None or _as_text(first_chapter.values[0]) != "1":
        raise ValueError(
            f"Expected chapter marker 1 at Excel row {first_chapter_excel_row}"
        )

    chapter_id = 0
    row_in_chapter = 0
    sequence_index = 0
    units: list[StimulusTextUnit] = []
    for row in material_rows:
        if row.excel_row == 1:
            continue
        text = _as_text(row.values[0] if row.values else None)
        if not text:
            continue
        is_marker = bool(_CHAPTER_MARKER.fullmatch(text))
        if is_marker:
            chapter_id = int(text)
            row_in_chapter = 0
        else:
            row_in_chapter += 1

        f1_label = _as_text(row.values[4]) if len(row.values) > 4 else ""
        m1_label = _as_text(row.values[5]) if len(row.values) > 5 else ""
        segment_role = "marker" if is_marker else f"row{row_in_chapter:04d}"
        segment_id = (
            f"{dataset_id}:{normalized_book}:ch{chapter_id:02d}:{segment_role}"
        )
        units.append(
            StimulusTextUnit(
                dataset_id=dataset_id,
                book_id=normalized_book,
                segment_id=segment_id,
                excel_row=row.excel_row,
                sequence_index=sequence_index,
                chapter_id=chapter_id,
                row_in_chapter=row_in_chapter,
                text=text,
                normalized_text=normalize_text(text),
                is_chapter_marker=is_marker,
                f1_boundary_label=f1_label or None,
                m1_boundary_label=m1_label or None,
            )
        )
        sequence_index += 1
    return units


def load_chineseeeg2_workbook(
    path: str | Path,
    *,
    book_id: str,
    first_chapter_excel_row: int,
) -> list[StimulusTextUnit]:
    sheet = read_xlsx_sheet(path)
    return parse_chineseeeg2_rows(
        sheet.rows,
        book_id=book_id,
        first_chapter_excel_row=first_chapter_excel_row,
    )

