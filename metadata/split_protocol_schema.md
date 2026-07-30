# Split protocol artifact schema

`split-protocol-v1` defines immutable, protocol-level partitions derived from
`metadata/all_trials.parquet`. These artifacts do not replace the manifest's
embedded split and do not encode window, feature, model, or retrieval state.

## Common top-level fields

- `protocol_version`: currently `split-protocol-v1`.
- `created_from_manifest_schema_version`: authoritative input manifest schema.
- `manifest_path`, `manifest_row_count`: input identity and complete row count.
- `manifest_critical_field_fingerprint`: SHA-256 over the physical manifest row
  order and the same critical fields used by the manifest audit.
- `manifest_embedded_split`: records that the seed-20260730 manifest split was
  preserved.
- `seed`, `requested_ratios`, `actual_ratios`: protocol seed and group-level
  requested/observed ratios. Trial ratios are descriptive and need not be 80/10/10.
- `setting`: protocol definition.
- `identity_keys`: `record_id` for trials, `split_group_id` for content, and a
  namespaced `subject_group_id` for people.
- `subject_namespace_method`: versioned construction from
  `dataset_version::paradigm::subject_id`; bare `subject_id` is never a
  cross-dataset identity.
- `deterministic_ordering`: assignment, list ordering, and JSON serialization
  contract.
- `key_assumptions_and_limitations`: scientific interpretation boundaries.

## Partition object

Every `train`, `validation`, and `test` member under `partitions` contains:

- sorted `record_ids`;
- sorted `split_group_ids`;
- sorted `subject_group_ids`;
- `trial_count`, `content_group_count`, `subject_group_count`;
- `paradigm_trial_counts`.

`excluded_record_ids` is sorted. `excluded_by_reason` maps each stable reason to
its sorted record IDs. A record occurs exactly once across all selected
partitions and exclusions. Settings with no exclusion use empty lists/maps.

`counts` reports manifest, selected, excluded, partition trial, content-group,
and subject-group counts. `leakage_checks` records content, subject, record-ID,
target-paradigm, and accounting invariants where applicable.

`content_assignment.split_group_ids` separately records the complete
group-level train/validation/test assignment, including groups whose trials are
all excluded by a strict diagonal or transfer cell. Partition-level
`split_group_ids` describe groups represented among selected trials.

## Setting A

`text_unseen_seed42.json` partitions each unique `split_group_id` once using
SHA-256 and seed 42. All trials inherit that assignment. Subjects may overlap
across partitions, and actual train/test overlap is explicit.

## Setting B

`subject_text_unseen_seed42.json` independently partitions content groups and
cohort-namespaced subject groups. Subject groups are SHA-256 ranked inside each
`dataset_version::paradigm` cohort, then assigned deterministic integer quotas
that keep validation and test non-empty for cohorts of at least three people.

Only diagonal cells are selected:

- train subject × train content;
- validation subject × validation content;
- test subject × test content.

All six off-diagonal cells are excluded with stable reason names of the form
`off_diagonal_subject_<partition>__content_<partition>`.

## Setting C

`cross_paradigm_seed42.json` contains three named zero-shot protocols under
`protocols`. Each protocol has the same partition/exclusion schema plus
`source_paradigms` and `target_paradigm`.

- train/validation contain only source-paradigm EEG;
- test contains only target-paradigm EEG;
- source test-content and target non-test-content cells are excluded;
- out-of-scope paradigms are excluded;
- whole content groups implicated by fuzzy global alignment, missing text, or
  uncertain material variants are excluded from strict main protocols.

No seen-content supplementary protocol is mixed into the main results.

## Deterministic serialization

JSON uses UTF-8, LF newlines, `sort_keys=True`, two-space indentation, and a
terminal newline. Identifier lists use ascending Unicode code-point order.
Artifacts contain no timestamps. Rebuilding from the same manifest, schema,
seed, and implementation must reproduce identical bytes and SHA-256.
