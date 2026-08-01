# Split Protocol Audit

This audit covers protocol-level assignments only. It does not generate windows, embeddings, models, or retrieval results.

## Immutable input contract

- Manifest: `metadata/all_trials.parquet`
- Rows: 168,156
- Critical-field SHA-256: `0afc05cf298ab5409ee8d4806edad0b164378ce1417c3024353cd22da3fceccf`
- Protocol seed: `42`
- The manifest's embedded split is preserved and is not overwritten.

## Artifact fingerprints

| artifact | SHA-256 |
|---|---|
| `cross_paradigm_seed42.json` | `806cd34b7fd9c1ae561db34c914b2d3c3608fce0937f2ac651040816fd774c01` |
| `subject_text_unseen_seed42.json` | `c760f8ee49f623dde7886187d147c13b45d7f3520170736afcb26b39a9d18470` |
| `text_unseen_seed42.json` | `559f25084bc02cc2f86375c393ccd71ad5b01102e62b6243bddd4627033ceaa1` |

## Setting A — unseen text, subjects visible

| protocol | train trials/groups/subjects | validation trials/groups/subjects | test trials/groups/subjects | excluded |
|---|---:|---:|---:|---:|
| A | 134,859/13,380/22 | 16,413/1,614/22 | 16,884/1,684/22 | 0 |

- Train/test content overlap: 0
- Train/test subject overlap (allowed): 22
- Content-group ratios (train/validation/test): 80.225%/9.677%/10.097%
- Trial ratios (train/validation/test): 80.199%/9.761%/10.041%

## Setting B — unseen subjects and unseen text

| protocol | train trials/groups/subjects | validation trials/groups/subjects | test trials/groups/subjects | excluded |
|---|---:|---:|---:|---:|
| B strict diagonal | 104,618/13,376/16 | 1,526/991/3 | 2,188/1,621/3 | 59,824 |

- Train/test content overlap: 0
- Train/test subject overlap: 0
- Off-diagonal excluded trials: 59,824
- Content-group assignment ratios (train/validation/test): 80.225%/9.677%/10.097%
- Selected-trial distribution (train/validation/test): 96.572%/1.409%/2.020%

### Subject cohort quotas

| cohort | subjects | train | validation | test |
|---|---:|---:|---:|---:|
| ChineseEEG1::silent_reading | 10 | 8 | 1 | 1 |
| ChineseEEG2::passive_listening | 8 | 6 | 1 | 1 |
| ChineseEEG2::reading_aloud | 4 | 2 | 1 | 1 |

### Off-diagonal exclusions

| reason | trials |
|---|---:|
| `off_diagonal_subject_test__content_train` | 17,442 |
| `off_diagonal_subject_test__content_validation` | 2,111 |
| `off_diagonal_subject_train__content_test` | 13,078 |
| `off_diagonal_subject_train__content_validation` | 12,776 |
| `off_diagonal_subject_validation__content_test` | 1,618 |
| `off_diagonal_subject_validation__content_train` | 12,799 |

## Setting C — zero-shot cross-paradigm, unseen text

| protocol | train trials/groups/subjects | validation trials/groups/subjects | test trials/groups/subjects | excluded |
|---|---:|---:|---:|---:|
| pl_to_silent_reading_unseen_text: passive_listening → silent_reading | 32,283/3,970/8 | 3,777/466/8 | 10,640/1,192/10 | 121,456 |
| silent_reading_to_pl_unseen_text: silent_reading → passive_listening | 85,944/9,612/10 | 10,685/1,179/10 | 4,129/518/8 | 67,398 |
| pl_silent_reading_to_ra_unseen_text: passive_listening,silent_reading → reading_aloud | 118,227/13,362/18 | 14,462/1,612/18 | 2,095/518/4 | 33,372 |

Selected-trial train/validation/test distributions:
- `pl_to_silent_reading_unseen_text`: 69.128%/8.088%/22.784%
- `silent_reading_to_pl_unseen_text`: 85.297%/10.605%/4.098%
- `pl_silent_reading_to_ra_unseen_text`: 87.716%/10.730%/1.554%

Every main protocol has zero target-paradigm trials in train/validation and zero train/test content overlap.

### Strict identity sensitivity masks

| mask | content groups | manifest trials in groups |
|---|---:|---:|
| `fuzzy_global_alignment_group_excluded_from_strict_main` | 15 | 300 |
| `missing_text_group_excluded_from_strict_main` | 6 | 21 |
| `material_variant_uncertain_group_excluded_from_strict_main` | 3 | 5 |

## Quality and alignment treatment

- `quality_filtering`: None. Protocol construction is independent of quality_flag.
- `fuzzy_global_alignment`: Retained in Settings A/B accounting and assignment; whole implicated content groups excluded from Setting C strict main protocols.
- `missing_text`: Retained in Settings A/B accounting and assignment; whole implicated content groups excluded from Setting C strict main protocols.
- `material_variant_uncertain`: Retained in Settings A/B accounting and assignment; whole implicated content groups excluded from Setting C strict main protocols.
- `implausible_eeg_trial_duration`: Retained everywhere unless excluded by a protocol cell/mask unrelated to duration; never used to assign a partition.
- `unresolved_global_alignment`: Retained. Zero-shot transfer does not require paired source/target trials; unresolved cross-dataset variants remain a scientific risk.

## Remaining scientific risks

- A deterministic content split prevents exact `split_group_id` leakage, but near-duplicate or semantically equivalent text not merged by the manifest can still cross partitions.
- Trial counts need not follow 80/10/10 because assignment is group-level and repeated content has unequal numbers of trials.
- Setting B's strict diagonal excludes most off-diagonal observations; reported performance therefore conditions on both held-out subject and held-out content quotas.
- Cross-paradigm cohorts are disjoint and acquisition/paradigm shifts are confounded with dataset and subject cohort.
- Unresolved global alignment is retained; strict conclusions should be paired with a sensitivity analysis that masks it.
- Protocol construction alone does not remove duration, position, padding, subject, or audio-envelope shortcuts; those controls belong to later evaluation stages.
