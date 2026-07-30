from __future__ import annotations

import zipfile
from pathlib import Path

from data.readers.simple_xlsx import XlsxRow, read_xlsx_sheet
from data.readers.stimulus_text import parse_chineseeeg2_rows


def _write_minimal_xlsx(path: Path) -> None:
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets><sheet name="Sheet" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1"
  Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
  Target="worksheets/sheet1.xml"/>
</Relationships>"""
    sheet = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>ChineseText</t></is></c></row>
  <row r="2"><c r="A2" t="inlineStr"><is><t>序言</t></is></c></row>
  <row r="3"><c r="A3" t="inlineStr"><is><t>1</t></is></c></row>
  <row r="4"><c r="A4" t="inlineStr"><is><t>你好。</t></is></c>
   <c r="E4" t="inlineStr"><is><t>f1ch1</t></is></c></row>
 </sheetData>
</worksheet>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def test_simple_xlsx_preserves_excel_rows_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "stimulus.xlsx"
    _write_minimal_xlsx(path)
    parsed = read_xlsx_sheet(path)
    assert parsed.name == "Sheet"
    assert parsed.max_column == 5
    assert parsed.rows[3].excel_row == 4
    assert parsed.rows[3].values == ("你好。", None, None, None, "f1ch1")


def test_stimulus_parser_assigns_chapter_and_segment_ids() -> None:
    rows = [
        XlsxRow(1, ("ChineseText",)),
        XlsxRow(2, ("序言",)),
        XlsxRow(3, ("1",)),
        XlsxRow(4, ("你好。", None, None, None, "f1ch1", None)),
    ]
    units = parse_chineseeeg2_rows(
        rows,
        book_id="littleprince",
        first_chapter_excel_row=3,
    )
    assert units[0].chapter_id == 0
    assert units[1].is_chapter_marker
    assert units[1].chapter_id == 1
    assert units[2].segment_id == "chineseeeg2:littleprince:ch01:row0001"
    assert units[2].f1_boundary_label == "f1ch1"

