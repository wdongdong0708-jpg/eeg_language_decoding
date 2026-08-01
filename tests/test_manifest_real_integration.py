import json
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from data.manifest import EEG_END_SAMPLE_SEMANTICS, manifest_arrow_schema
from data.trial_manifest import (
    GARNETTDREAM_AUDIO_ALIGNMENT_VERSION,
    load_ce2_catalog,
    load_validated_garnettdream_audio_spans,
    select_ce2_run_units,
)


MANIFEST = Path("metadata/all_trials.parquet")
DIAGNOSTICS = Path("metadata/manifest_build_diagnostics.json")
MATERIAL_ROOT = Path(r"D:\dataset\ChineseEEG-2\materials&embeddings")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def table():
    if not MANIFEST.is_file():
        pytest.skip("Real manifest has not been built")
    return pq.read_table(MANIFEST)


def test_real_manifest_matches_all_legal_pairs_minus_chapter_markers(table) -> None:
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    assert diagnostics["excluded_chapter_marker_pairs"] == 810
    assert table.num_rows + diagnostics["excluded_chapter_marker_pairs"] == 168966
    assert diagnostics["paradigm_counts"] == {
        "passive_listening": 40302,
        "reading_aloud": 20440,
        "silent_reading": 107414,
    }


def test_real_manifest_schema_and_half_open_intervals(table) -> None:
    assert table.schema.equals(manifest_arrow_schema(), check_metadata=True)
    assert pc.all(
        pc.greater(table["eeg_end_sample"], table["eeg_start_sample"])
    ).as_py()
    assert set(table["eeg_end_sample_semantics"].to_pylist()) == {
        EEG_END_SAMPLE_SEMANTICS
    }


def test_audio_null_and_evidence_policies(table) -> None:
    rows = table.select(
        [
            "dataset_version",
            "paradigm",
            "book_id",
            "global_text_id",
            "audio_file",
            "audio_start_sec",
            "audio_end_sec",
            "audio_alignment_method",
        ]
    ).to_pylist()
    assert all(
        row["audio_file"] is None
        for row in rows
        if row["dataset_version"] == "ChineseEEG1"
    )
    assert all(
        row["audio_start_sec"] is None and row["audio_end_sec"] is None
        for row in rows
        if row["paradigm"] == "reading_aloud"
    )
    garnettdream_pl = [
        row
        for row in rows
        if row["paradigm"] == "passive_listening"
        and row["book_id"] == "garnettdream"
    ]
    assert all(
        row["audio_start_sec"] is not None
        and row["audio_alignment_method"]
        == GARNETTDREAM_AUDIO_ALIGNMENT_VERSION
        for row in garnettdream_pl
        if row["global_text_id"] is not None
    )
    assert any(
        row["audio_start_sec"] is not None
        for row in rows
        if row["paradigm"] == "passive_listening"
        and row["book_id"] == "littleprince"
    )


def test_garnettdream_audio_uses_two_acquisition_segmentations() -> None:
    if not MATERIAL_ROOT.is_dir():
        pytest.skip("ChineseEEG2 materials are unavailable")
    catalog = load_ce2_catalog(MATERIAL_ROOT)
    spans, diagnostics = load_validated_garnettdream_audio_spans(
        MATERIAL_ROOT / "audio",
        catalog,
    )
    assert diagnostics["f1"]["validated"] is True
    assert diagnostics["m1"]["validated"] is True
    assert {
        run_id: details["event_pair_count"]
        for run_id, details in diagnostics["f1"]["runs"].items()
    } == {
        "11": 232,
        "12": 263,
        "13": 169,
        "14": 341,
        "15": 294,
        "21": 197,
        "22": 282,
        "23": 235,
        "24": 134,
    }
    assert {
        run_id: details["event_pair_count"]
        for run_id, details in diagnostics["m1"]["runs"].items()
    } == {
        "11": 494,
        "12": 169,
        "13": 341,
        "14": 490,
        "21": 282,
        "22": 235,
        "23": 294,
        "24": 186,
    }
    assert len(spans) == 4638
    assert not any(speaker == "m1" and run_id == "15" for speaker, run_id, _ in spans)

    f1_run11, _, _ = select_ce2_run_units(
        book_id="garnettdream",
        run_id="11",
        subject_id="01",
        paradigm="passive_listening",
        catalog=catalog,
    )
    assert (f1_run11[0].source_excel_row, f1_run11[-1].source_excel_row) == (
        23,
        254,
    )
    row_1320 = next(
        unit for unit in catalog["garnettdream"] if unit.source_excel_row == 1320
    )
    assert spans[("f1", "15", row_1320.global_text_id)].audio_file.endswith(
        "audio_5.wav"
    )
    assert spans[("f1", "21", row_1320.global_text_id)].audio_file.endswith(
        "audio_6.wav"
    )


def test_same_split_group_never_crosses_splits(table) -> None:
    by_group: dict[str, set[str]] = defaultdict(set)
    for group_id, split in zip(
        table["split_group_id"].to_pylist(),
        table["split"].to_pylist(),
    ):
        by_group[group_id].add(split)
    assert all(len(splits) == 1 for splits in by_group.values())


def test_accepted_fuzzy_global_variants_share_split(table) -> None:
    by_global: dict[str, set[str]] = defaultdict(set)
    for row in table.select(
        [
            "global_text_id",
            "global_text_alignment_status",
            "split_group_id",
        ]
    ).to_pylist():
        if row["global_text_alignment_status"] == "fuzzy":
            by_global[row["global_text_id"]].add(row["split_group_id"])
    assert by_global
    assert all(len(groups) == 1 for groups in by_global.values())
