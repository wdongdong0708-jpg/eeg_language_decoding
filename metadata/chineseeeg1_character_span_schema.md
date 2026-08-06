# ChineseEEG1 fixed-character-span schema

Schema version: `ce1-fixed-character-span-v1`.

The local Parquet artifact is
`metadata/generated/chineseeeg1_character_spans_seed42.parquet`. It is generated
from `metadata/all_trials.parquet`, the Setting-A seed-42 protocol artifact, and
the character-timeline audit. Parquet files are intentionally Git-ignored; the
versioned JSON/Markdown audit contains its SHA-256.

## Identities and partition inheritance

- `record_id`, `block_id`, `content_id`, `split_group_id`: inherited Stage-1
  identities.
- `split`: inherited from `splits/text_unseen_seed42.json`; it is never computed
  from a subject, row index, span index, or DataLoader order.
- `global_text_id`: source stimulus-row text identity.
- `span_event_id`: hash of `global_text_id` plus half-open presentation-clock
  offsets. It is the same across subjects viewing the same local event.
- `span_text_id`: hash of the clock-character string only. Identical local text
  in different parent rows has the same identity and is a multi-positive/false-
  negative case, not an ordinary negative.
- `subject_group_id`: cohort-namespaced subject identity; it is metadata and is
  not an input to the primary no-subject-layer model.

Every `split_group_id` and every `span_event_id` occurs in exactly one partition.
The primary protocol guarantees unseen parent text blocks. Because 4/6/8-character
strings can recur, evaluation also exposes a strict-unseen-local-string subset.

## Text offsets

- `source_sentence_text`: the exact released stimulus display row. It is not a
  linguistic word and must not be described as a word-timestamp unit.
- `preceding_context_text`, `following_context_text`: neighboring display-row
  context within the same recording; neither is passed to the EEG model.
- `span_text`: exactly 4, 6, or 8 characters that advance the official visual
  clock. Program-skipped punctuation is omitted.
- `span_surface_text`: raw substring from the first through last clock character;
  it preserves intervening punctuation.
- `span_start_char`, `span_end_char`: half-open Python string offsets into
  `source_sentence_text`.
- `span_start_clock`, `span_end_clock`: half-open offsets into the sequence of
  official presentation-clock characters.
- `span_position_fraction`, `stimulus_position`: stored for shortcut audits only;
  the primary model must not receive them.

Tokenizer alignment uses fast-tokenizer `offset_mapping`. A character is mapped
to every non-special token whose raw interval overlaps the character interval.
Token sequence positions are never guessed from character positions.

## EEG bounds and fixed model length

- `source_row_start_sample`, `source_row_stop_sample`: inherited `ROWS`/`ROWE`
  half-open block.
- `eeg_start_sample`, `eeg_stop_sample`: approximate visual span after configured
  neural delay and left/right context.
- `source_eeg_sample_count`: source samples before resampling. It is retained for
  provenance and shortcut evaluation, not passed to the model.
- `model_eeg_sample_count`: fixed per span length: 358 (4 chars), 538 (6 chars),
  or 717 (8 chars) at 256 Hz under the current configuration.
- `padding_samples`: always zero.
- `exposes_padding_mask`: always false.
- `resampling_method`: linear interpolation to the fixed span-length target.

If neural delay or context would cross the source row, the span is dropped. It is
never clipped, padded, or allowed to cross into another content/split block.

## Timing provenance

- `timeline_method`: primary index uses `event_affine`.
- `timeline_source`: pinned official presentation code plus the released
  `ROWS`/`ROWE` pair.
- `timeline_rule`: affine subdivision of the observed row interval by the exact
  official clock-character count.
- `effective_character_interval_sec`: observed row duration divided by count.
- `configured_clock_boundary_disagreement_sec`: maximum boundary difference
  between event-affine timing and the configured 0.35-second schedule.
- `alignment_confidence`: `medium` or `low`; the primary artifact excludes low
  confidence.
- `exact_character_onsets_observed`: always false.

These fields describe an approximate visual time axis. They are not word or
speech timestamps.

## Frozen text representations

The extractor supports local-only encoding and full-source-row encoding. Cached
files retain token IDs, special-token flags, raw offsets, token states, character
states, and character-to-token mappings. Supported target construction includes:

- offset-selected character/token mean;
- learned attention pooling over a fixed character sequence;
- local or full-row `[CLS]` controls;
- arbitrary contextual hidden layers;
- a true static baseline using the frozen model input-token embedding table
  (`representation_source=input_token_embedding`, `layer_index=0`), without
  running contextual Transformer layers.

## False negatives

Pair classification distinguishes random, same-length, high lexical overlap,
semantic-near/different-event, and adjacent-text negatives. Same event, identical
local text, and highly overlapping spans are false negatives. The configured
policy either masks them from the denominator or assigns explicit soft-positive
weight; the contrastive loss rejects any policy that masks a positive.
