# PL EEG–speech baseline validation

This is an engineering/synchronization baseline. It is not evidence of semantic or linguistic decoding.

## Data and windows

- PL manifest trials: 40,302
- Exact 3-second windows: 2,677
- Train/validation/test windows: 2,173/209/295
- Unique speech targets: 688
- Content/audio-target cross-split leakage: 0/0

## Audio targets

- Model: `airesearch/wav2vec2-large-xlsr-53-th`
- Hidden layers averaged: `[14, 15, 16, 17, 18]`
- Temporal pooling: false
- Output shape: `[1024, 750]`
- Cached targets: 688

## Delay sweep

- Common support: 1,217 windows
- Delays: 0/100/200/300/400/500 ms
- Provisional validation-selected delay: 500 ms
- Identical epoch-0 batch schedule across delays: true
- Test EEG was not used during delay selection.

## One-time selected-delay test

- Queries/candidates: 150/38

| pool | model R@1 | model R@10 | random R@1 | random R@10 | position-only R@1 | position-only R@10 |
|---|---:|---:|---:|---:|---:|---:|
| global | 0.0267 | 0.3400 | 0.0133 | 0.2067 | 0.0333 | 1.0000 |
| position_local | 0.0600 | 0.5667 | 0.0200 | 0.4333 | 0.0333 | 1.0000 |

## Scientific boundary

Training loss and retrieval above random confirm that the pipeline can learn and rank synchronized targets. They do not isolate linguistic content. Sentence position reaches R@10=1.0 on the selected common-support test pool, character count is also strong, and the untrained envelope baseline is competitive on some metrics. This dataset subset should therefore remain an engineering unit test unless stronger controls or additional independently ordered audio material become available.

The seed-42 test set is not pristine for confirmatory inference: its shortcut diagnostics were inspected while the evaluation protocol was being hardened. The selected checkpoint score remains useful as an engineering check, but a confirmatory claim requires a newly predeclared split/seed or untouched external cohort.
