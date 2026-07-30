# Initial data and official-code audit

Audit date: 2026-07-30

## Scope

Read-only inspection covered the official ChineseEEG-2 repository at commit
`b1c4ba8afd738383cf5d4676f65256aa43d876e3` and the local preprocessed
datasets. No data were downloaded, rewritten or reprocessed.

## Official implementation findings

- `novel_segmentation/cut_chinese_novel.py` constructs display rows and repeats
  each display item once per non-punctuation character. The RA default
  `shift_time` is 0.25 s.
- RA `ROWS`/`ROWE` denote the beginning/end of a highlighted screen row, not a
  linguistic word and not a measured spoken-word boundary.
- RA audio is recorded as chapter/session WAV files. The inspected local audio
  is stereo, 12 kHz, 16-bit.
- PL plays those recorded WAV files and regenerates `ROWS`/`ROWE` from
  `events_data.json` after converting original EEG sample indices to audio
  time. This is useful for row alignment but is not a forced alignment.
- Official text embeddings use `bert-base-chinese` and
  `last_hidden_state.mean(dim=1)`.
- Official audio embeddings use
  `airesearch/wav2vec2-large-xlsr-53-th` and
  `last_hidden_state.mean(dim=1)`.
- Official preprocessing code performs segmentation, downsampling, filtering,
  bad-channel interpolation, ICA and average reference. This project consumes
  the supplied derivatives and does not rerun that pipeline by default.

## Local data organization

### ChineseEEG1

- The task-provided path
  `D:/dataset/ChineseEEG/filtered_0.5_30` is absent.
- The available data are at
  `D:/dataset/ChineseEEG/derivatives/preproc/filtered_0.5_30`.
- Ten subject directories are present:
  `sub-04`–`sub-10` and `sub-13`–`sub-15`.
- BrainVision headers report 128 channels and 256 Hz.
- LittlePrince uses run IDs 01–07; GarnettDream uses 01–19, with documented
  missing/substituted runs.
- Each non-display run XLSX row count matches its provided `(N, 768)` embedding
  row count for the ordinary runs inspected. GarnettDream run 19 has stimulus
  XLSX files but no corresponding official embedding file.

### ChineseEEG2 Passive Listening

- Eight subjects (`sub-01`–`sub-08`) and 284 BrainVision recordings are present.
- BrainVision headers report 128 channels and 250 Hz.
- Five recordings have neither `events.tsv` nor marker entries in their
  BrainVision `.vmrk` file and therefore cannot yield row blocks:
  - `sub-04`, LittlePrince runs `27`, `28`, `29`, `210`
  - `sub-05`, LittlePrince run `114`
- LittlePrince uses encoded run IDs
  `11..19, 110..114, 21..29, 210..213`.
- GarnettDream uses `11..15, 21..24`.
- Local documentation maps PL subjects 01–04 to speaker `f1` and 05–08 to
  speaker `m1`.

### ChineseEEG2 Reading Aloud

- Four subjects (`sub-f1`, `sub-f2`, `sub-m1`, `sub-m2`) and 144 BrainVision
  recordings are present.
- BrainVision headers report 128 channels and 250 Hz.
- Run IDs follow the same encoded scheme as PL.

### ChineseEEG2 materials and embeddings

- LittlePrince material has 2,854 non-empty text rows.
- GarnettDream material has speaker-specific boundary annotations:
  nine `f1` sections in column E and eight `m1` sections in column F.
- PL subjects 01–04 and 05–08 therefore cannot be assumed to share identical
  GarnettDream row-to-run segmentation.
- `events_data.json` contains chapter/audio start indices plus row start/end
  sample indices. GarnettDream files contain a one-event ROWS/ROWE count
  mismatch, which must be resolved explicitly during manifest audit.
- Provided arrays inspected:
  - LittlePrince text: `(2853, 768)`
  - GarnettDream text: `(2164, 768)`
  - LittlePrince f1 sentence audio: `(2853, 1024)`
  - LittlePrince f1 frame audio: `(267308, 1024)`

The differing material/embedding lengths must not be joined by row index without
an explicit validation report.

## Consequences for the implementation

1. Preserve `book_id`, canonical sentence/row sequence, speaker/material
   variant and source file in the manifest.
2. Keep `block_id` (a concrete recording occurrence) distinct from
   `content_id` (subject/paradigm-independent stimulus identity).
3. Derive split from `content_id` before window generation.
4. Treat `ROWS`/`ROWE` as row-level evidence. Spoken word/character timing
   requires forced alignment, ASR timestamps or a documented weak monotonic
   alignment.
5. Fail on row-count or boundary mismatches rather than truncating with `zip`
   or joining by position.
