# Feature artifact schema

The feature extractors write compressed NumPy archives (`.npz`) with
`allow_pickle=False` compatibility. Every artifact contains a JSON metadata scalar
plus numeric/string arrays. Raw EEG, WAV, and stimulus files are never modified.

## Text

Implementation: `src/features/text_features.py`

Default model: `bert-base-chinese`

One artifact corresponds to one canonical `content_id` and stores:

| field | shape | meaning |
|---|---|---|
| `sentence_hidden_state` | `[hidden]` | sentence representation |
| `token_indices` | `[tokens]` | tokenizer sequence positions, including retained special tokens |
| `token_ids` | `[tokens]` | tokenizer vocabulary IDs |
| `tokens` | `[tokens]` | tokenizer strings |
| `token_offsets` | `[tokens, 2]` | half-open character offsets into the exact input string |
| `special_tokens_mask` | `[tokens]` | identifies `[CLS]`, `[SEP]`, and other special tokens |
| `token_hidden_states` | `[tokens, hidden]` | unpooled hidden states |
| `character_indices` | `[characters]` | indices into the exact input string |
| `characters` | `[characters]` | non-whitespace Unicode characters |
| `character_offsets` | `[characters, 2]` | half-open source character offsets |
| `character_hidden_states` | `[characters, hidden]` | mean of token states overlapping each character |
| `character_is_highlighted` | `[characters]` | excludes punctuation from the fixed visual-highlight clock |
| `character_token_indptr`, `character_token_indices` | CSR | complete character-to-token provenance |

`mean_content_tokens` is the research default: it excludes padding and special
tokens. `mean_attended_tokens` reproduces the official per-sentence
`last_hidden_state.mean(dim=1)` result while remaining invariant to batch padding:
the official script processes one sentence at a time, so all sequence positions are
attended. `cls` is available as an explicit alternative.

Fast-tokenizer offset mappings are mandatory. The extractor raises on unmapped
non-whitespace characters by default and records truncation. Character states are
derived only from tokenizer offsets; they do not assert word boundaries or timing.

With `local_files_only=True`, the loader enables strict Hugging Face offline mode,
resolves the requested model ID to a cached snapshot directory, and passes that
directory to Transformers. This prevents PEFT adapter discovery from making an
unexpected HTTP request. A missing or incomplete cache fails without a network
fallback.

## Audio

Implementation: `src/features/audio_features.py`

Default model: `airesearch/wav2vec2-large-xlsr-53-th`

One artifact corresponds to one already split audio `block_id` and stores:

| field | shape | meaning |
|---|---|---|
| `frame_indices` | `[frames]` | wav2vec output-frame index |
| `frame_offsets_sec` | `[frames, 2]` | half-open receptive-field span on the source-file timeline |
| `frame_hidden_states` | `[frames, hidden]` | unpooled wav2vec hidden states |

Metadata preserves `block_id`, `content_id`, inherited `split`, source path,
source/model sample rates, quantized source start/stop samples and seconds, layer,
convolution stride, and receptive field. The loader reads only the requested block
before resampling and feature extraction. A frame stop is clipped to the block stop,
so an artifact never claims support outside its split boundary.

The optional `mean_pool_audio_frames` function can reproduce whole-block or
sub-span mean pooling without discarding the frame-level artifact.

## Timing semantics

- Character offsets are textual offsets, not time.
- `character_is_highlighted` may be combined with the validated 0.35 s
  ChineseEEG1 or 0.25 s ChineseEEG2 visual-presentation clock.
- That clock must not be used as RA pronunciation timing.
- PL/RA spoken spans require event evidence, forced alignment, ASR timestamps, or
  a separately documented weak-alignment procedure.
- Every downstream EEG/audio span must remain inside one manifest block and inherit
  that block's deterministic content split.
