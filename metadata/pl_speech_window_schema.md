# PL EEG–speech window schema

`pl-speech-window-v1` is a model-input index derived from the immutable trial
manifest and `text_unseen_seed42.json`. It does not alter either source.

Each JSONL row is one fixed-duration physical-time pairing:

- `split_group_id` and `split` inherit Setting A unchanged;
- `eeg_start_sample:eeg_stop_sample` is a half-open EEG range;
- `audio_start_sample:audio_stop_sample` is the paired half-open wav range;
- `eeg_delay_ms` means the EEG span begins that many milliseconds after the
  corresponding audio-relative start;
- both spans remain inside the same manifest trial block;
- `audio_target_id` is shared across subjects only when source wav identity and
  exact source-sample bounds match;
- `window_id` includes record, audio target, EEG samples, and configured delay.

The first protocol uses 3-second, 3-second-stride, drop-tail windows. Thus every
selected row has 750 valid EEG samples at 250 Hz and zero padded samples.

Rows without verified audio alignment, rows whose audio boundary exceeds the
actual wav file, and trials too short after applying the delay are retained in
the companion audit under explicit exclusion reasons.

Shortcut-control metadata (`subject_group_id`, `stimulus_position`,
`char_count`, source trial duration, valid/padded counts) is present for audit
and baseline evaluation only. It is not passed to the EEG encoder.

Window creation is deterministic. JSON keys and rows are sorted, and rebuilding
from the same manifest, split artifact, configuration, and source file metadata
must reproduce the same SHA-256.
