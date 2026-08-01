# Trial Manifest Audit

This report audits complete stimulus-row trials. `sentence_id` denotes an experimental stimulus row/short segment; it is not asserted to be a linguistic sentence or a word-aligned unit.

## Contract

- Only legal `ROWS` followed by `ROWE` pairs are emitted.
- EEG intervals use half-open `[eeg_start_sample, eeg_end_sample)` bounds; `eeg_end_sample` is the exclusive ROWE event-onset sample.
- Chapter-marker pairs are excluded from trials and retained only as context.
- Raw workbook text is preserved; normalization is separate and traced.
- RA screen events are never used as spoken-audio boundaries.
- Split assignment is SHA-256 over versioned content groups with seed `[20260730]`.

## Manifest

- Rows: 168,156
- Columns: 58
- Unique primary keys: `True`
- Authoritative schema match: `True`
- Excluded chapter-marker event pairs: 810

### Dataset/paradigm counts

| dataset | paradigm | trials |
|---|---|---:|
| ChineseEEG1 | silent_reading | 107,414 |
| ChineseEEG2 | passive_listening | 40,302 |
| ChineseEEG2 | reading_aloud | 20,440 |

The full dataset/paradigm/subject/book/chapter table is available in `reports/manifest_audit.json`.

## Missingness and event integrity

- Trials with unresolved local text: 21
- Recordings missing event TSV: 5
- PL trials without validated audio bounds: 33
- RA trials with intentionally null audio bounds: 20,440

### Source event anomalies

| source | missing events recordings | orphan ROW markers | events/vmrk mismatch recordings | broken references |
|---|---:|---:|---:|---:|
| chineseeeg1 | 0 | 8 | 0 | 172 |
| chineseeeg2_pl | 5 | 30 | 1 | 0 |
| chineseeeg2_ra | 0 | 6 | 0 | 0 |

## Alignment

- Event-to-source statuses: `{"exact": 165189, "fuzzy": 2946, "unresolved": 21}`
- Cross-dataset/global statuses: `{"exact": 63118, "fuzzy": 131, "unresolved": 104907}`
- `global_text_id` conflicts: 0

ChineseEEG1-to-ChineseEEG2 sharing uses monotonic exact anchors and reciprocal fuzzy matches inside anchor gaps. It never joins by row number across datasets. Unaccepted matches retain dataset-local IDs.

## Quality flags

| flag | trials |
|---|---:|
| broken_brainvision_reference | 86,264 |
| event_text_count_mismatch | 2,951 |
| events_vmrk_count_mismatch | 41 |
| implausible_eeg_trial_duration | 8 |
| material_variant_uncertain | 5 |
| missing_text | 21 |
| ok | 38,927 |
| orphan_row_event_in_recording | 6,301 |
| pl_audio_mapping_unverified | 33 |
| prechapter_trial_unresolved | 14 |
| ra_audio_boundary_unavailable | 20,440 |
| unresolved_text_alignment | 104,907 |

## Deterministic splits and leakage

- Content groups: 16,678
- Group counts: `{"test": 1647, "train": 13385, "valid": 1646}`
- Trial counts: `{"test": 16730, "train": 135018, "valid": 16408}`
- Same `split_group_id` crossing splits: **0**
- Stored split versus fixed-hash mismatch: **0**

## Duration and shortcut-relevant fields

- EEG seconds: `{"count": 168156, "min": 0.004, "p05": 0.75390625, "median": 3.708, "p95": 4.3359375, "max": 5.188, "mean": 3.0840226986027854}`
- Validated audio seconds: `{"count": 40269, "min": 0.06400000000002137, "p05": 0.47999999999999987, "median": 1.9720000000000013, "p95": 4.200000000000003, "max": 5.188000000000002, "mean": 2.144952097146688}`

`char_count`, `raw_char_count`, `highlight_char_count`, EEG/audio duration and padding-relevant boundaries are explicit manifest fields for later shortcut baselines and length-matched candidate pools.

## Normalization and duplicate diagnostics

- Rule trial hits: `{"fullwidth_ascii_to_ascii": 94706, "punctuation_variant_canonicalization": 17570}`
- Exact repeated normalized-text groups: 446
- Near-duplicate diagnostic pairs: 2
- Split groups covered by all three paradigms: 295

Near-duplicate detection is audit-only and never merges identities.

## Parquet/CSV consistency

- Row count equal: `True`
- Primary-key order equal: `True`
- Critical-field digest equal: `True`
- Digest: `0afc05cf298ab5409ee8d4806edad0b164378ce1417c3024353cd22da3fceccf`

## Conservative unresolved mappings

- Grouped unresolved text entries: 13
- Grouped null audio entries: 177
- Recordings with no legal trial pair: 5

The complete grouped lists and source recording paths are in `reports/manifest_audit.json`. Null values are intentional wherever the evidence does not support an alignment.
