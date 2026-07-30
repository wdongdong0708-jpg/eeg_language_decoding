# PL EEG–speech window audit

- Schema: `pl-speech-window-v1`
- Manifest PL records: 40,302
- Records with windows: 2,677
- Excluded records: 37,625
- Windows: 2,677
- Unique audio targets: 688
- Content groups: 346
- Subject groups: 8

## Partition counts

| partition | windows |
|---|---:|
| train | 2,173 |
| validation | 209 |
| test | 295 |

## Exclusions

| reason | records |
|---|---:|
| `audio_bounds_exceed_file` | 179 |
| `shorter_than_window_after_delay` | 18,924 |
| `unverified_or_missing_audio_alignment` | 18,522 |

## Integrity

- Content groups crossing partitions: 0
- Audio targets crossing partitions: 0
- Duplicate window IDs: 0
- Complete record accounting: `True`
- All windows exact length: `True`

Only verified audio-aligned PL trials are eligible. Other quality flags are recorded but are not used as filters.
