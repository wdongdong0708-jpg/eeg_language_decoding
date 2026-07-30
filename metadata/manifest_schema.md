# Complete stimulus-row trial manifest schema

Schema version: `trial-manifest-v1`

One row is one complete stimulus trial defined by a legal `ROWS` event followed
by a later `ROWE` event in the same recording. The EEG interval is half-open:
`[eeg_start_sample, eeg_end_sample)`. `eeg_end_sample` is the exclusive sample
at the `ROWE` event onset. No event boundary is inferred or repaired.

`sentence_id` identifies an experimental stimulus row/short segment. It does
not assert a linguistic sentence, Chinese word, or word-timestamp unit.

## Identity and split fields

- `sentence_id`: versioned source stimulus-row identity (`sentence-v1`).
- `global_text_id`: versioned reviewed cross-dataset identity
  (`global-text-v1`). ChineseEEG1 uses monotonic sequence evidence; unresolved
  rows retain dataset-local global IDs.
- `text_hash`: SHA-256 of `normalized_text`; `raw_text_hash` and
  `normalized_text_hash` are both retained.
- `split_group_id`: `split-group-v1`, based on book plus reviewed identity
  normalized text. Exact repeated content and accepted monotonic variants share
  a group. Unresolved event text uses a conservative book/material quarantine
  group.
- `split`: deterministic SHA-256 assignment with seed `20260730`.
- `content_id`: versioned subject/session/paradigm-independent block content
  identity (`content-v2`).
- `record_id`: unique event trial identity (`trial-record-v1`).
- `block_id`: complete-trial block identity (`trial-block-v1`).

## Text policy

`raw_text` is the exact source-cell string and is never silently stripped or
rewritten. `normalized_text` follows `metadata/normalization_rules.json`;
`normalization_trace` is a JSON list of every rule and hit count. Simplified and
traditional Chinese are not converted.

`char_count` excludes Unicode whitespace but includes punctuation.
`raw_char_count` counts all Unicode codepoints. `highlight_char_count` excludes
Unicode punctuation, separators and control/format characters and represents
the experiment's visual character clock. `word_count` uses a pinned jieba
precise-mode dictionary method recorded in `word_count_method`; it is not a
BERT WordPiece count.

## Audio policy

- ChineseEEG1 audio fields are null.
- RA audio fields are null unless future forced-alignment/ASR evidence is
  reviewed; screen `ROWS/ROWE` events are not speech boundaries.
- PL boundaries are populated only for evidence-validated speaker/material
  audio rows. The evidence chain is stored in `audio_alignment_evidence`.

## Quality flags

Multiple flags are sorted and joined by `|`. The formal enum is
`data.manifest.QualityFlag`. Important values include:

- `missing_events_tsv`
- `orphan_row_event_in_recording`
- `events_vmrk_count_mismatch`
- `broken_brainvision_reference`
- `implausible_eeg_trial_duration`
- `event_text_count_mismatch`
- `missing_text`
- `unresolved_text_alignment`
- `prechapter_trial_unresolved`
- `ra_audio_boundary_unavailable`
- `pl_audio_mapping_unverified`

Rows with uncertain mappings remain in the manifest with null derived fields
and flags. Chapter-marker pairs are not emitted as trials but are counted in
the manifest audit.
