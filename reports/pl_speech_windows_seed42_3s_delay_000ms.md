# PL EEG–speech window audit

- Schema: `pl-speech-window-v1`
- Manifest PL records: 40,302
- Records with windows: 7,565
- Excluded records: 32,737
- Windows: 7,565
- Unique audio targets: 1,910
- Content groups: 999
- Subject groups: 8

## Partition counts

| partition | windows |
|---|---:|
| train | 5,977 |
| validation | 737 |
| test | 851 |

## Exclusions

| reason | records |
|---|---:|
| `audio_bounds_exceed_file` | 179 |
| `shorter_than_window_after_delay` | 32,525 |
| `unverified_or_missing_audio_alignment` | 33 |

## Integrity

- Content groups crossing partitions: 0
- Audio targets crossing partitions: 0
- Duplicate window IDs: 0
- Complete record accounting: `True`
- All windows exact length: `True`

Only verified audio-aligned PL trials are eligible. Other quality flags are recorded but are not used as filters.
