"""Minimal read-only XLSX reader for simple stimulus tables.

The ChineseEEG stimulus workbooks contain plain cell values and no formulas
needed by the research pipeline. This reader deliberately supports that narrow
surface using only the Python standard library, so manifest construction does
not depend on Excel application state.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REFERENCE = re.compile(r"^([A-Z]+)([0-9]+)$")


@dataclass(frozen=True, slots=True)
class XlsxRow:
    excel_row: int
    values: tuple[str | int | float | bool | None, ...]


@dataclass(frozen=True, slots=True)
class XlsxSheet:
    name: str
    rows: tuple[XlsxRow, ...]
    max_column: int


def column_index(column_letters: str) -> int:
    """Convert Excel column letters to a zero-based column index."""

    result = 0
    for character in column_letters:
        if not ("A" <= character <= "Z"):
            raise ValueError(f"Invalid Excel column: {column_letters!r}")
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    values: list[str] = []
    for item in root.findall(f"{{{_MAIN_NS}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")))
    return values


def _sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{_PKG_REL_NS}}}Relationship")
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.find(f"{{{_MAIN_NS}}}sheets") or []:
        relationship_id = sheet.attrib[f"{{{_REL_NS}}}id"]
        target = targets[relationship_id].lstrip("/")
        if not target.startswith("xl/"):
            target = posixpath.normpath(posixpath.join("xl", target))
        sheets.append((sheet.attrib["name"], target))
    return sheets


def _cell_value(
    cell: ElementTree.Element,
    shared_strings: list[str],
) -> str | int | float | bool | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_MAIN_NS}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{_MAIN_NS}}}t"))

    value_node = cell.find(f"{{{_MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    return int(numeric) if numeric.is_integer() else numeric


def read_xlsx_sheet(path: str | Path, *, sheet_index: int = 0) -> XlsxSheet:
    """Read a value-only worksheet while preserving original Excel row numbers."""

    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)

    with zipfile.ZipFile(workbook_path) as archive:
        sheets = _sheet_targets(archive)
        if not 0 <= sheet_index < len(sheets):
            raise IndexError(
                f"sheet_index={sheet_index} outside workbook with {len(sheets)} sheets"
            )
        sheet_name, sheet_path = sheets[sheet_index]
        shared_strings = _shared_strings(archive)
        root = ElementTree.fromstring(archive.read(sheet_path))

    parsed_rows: list[tuple[int, dict[int, object]]] = []
    max_column = 0
    sheet_data = root.find(f"{{{_MAIN_NS}}}sheetData")
    if sheet_data is None:
        return XlsxSheet(name=sheet_name, rows=(), max_column=0)

    for row in sheet_data.findall(f"{{{_MAIN_NS}}}row"):
        excel_row = int(row.attrib["r"])
        values: dict[int, object] = {}
        for cell in row.findall(f"{{{_MAIN_NS}}}c"):
            reference = cell.attrib.get("r", "")
            match = _CELL_REFERENCE.match(reference)
            if match is None:
                raise ValueError(f"Invalid cell reference in {workbook_path}: {reference}")
            index = column_index(match.group(1))
            values[index] = _cell_value(cell, shared_strings)
            max_column = max(max_column, index + 1)
        parsed_rows.append((excel_row, values))

    rows = tuple(
        XlsxRow(
            excel_row=excel_row,
            values=tuple(values.get(index) for index in range(max_column)),
        )
        for excel_row, values in parsed_rows
    )
    return XlsxSheet(name=sheet_name, rows=rows, max_column=max_column)

