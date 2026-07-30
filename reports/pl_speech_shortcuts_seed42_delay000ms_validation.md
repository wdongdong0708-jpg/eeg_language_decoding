# PL speech shortcut baselines

- Partition: `validation`
- Queries: 209
- Unique candidates: 53
- Tie policy: pessimistic

| baseline | pool | R@1 | R@5 | R@10 | median rank | MRR |
|---|---|---:|---:|---:|---:|---:|
| `random` | global | 0.0287 | 0.1435 | 0.2249 | 25.0 | 0.1056 |
| `random` | position_local | 0.0622 | 0.2775 | 0.5455 | 10.0 | 0.1999 |
| `duration_only` | global | 0.0000 | 0.0000 | 0.0000 | 53.0 | 0.0189 |
| `duration_only` | position_local | 0.0000 | 0.0000 | 0.0000 | 20.0 | 0.0500 |
| `padding_mask_only` | global | 0.0000 | 0.0000 | 0.0000 | 53.0 | 0.0189 |
| `padding_mask_only` | position_local | 0.0000 | 0.0000 | 0.0000 | 20.0 | 0.0500 |
| `character_count_only` | global | 0.0191 | 0.2488 | 0.6555 | 10.0 | 0.1893 |
| `character_count_only` | position_local | 0.0191 | 0.5407 | 1.0000 | 4.0 | 0.2946 |
| `sentence_position_only` | global | 0.0383 | 1.0000 | 1.0000 | 2.0 | 0.4809 |
| `sentence_position_only` | position_local | 0.0383 | 1.0000 | 1.0000 | 2.0 | 0.4809 |
| `subject_id_only` | global | 0.0000 | 0.0000 | 0.0000 | 27.0 | 0.0377 |
| `subject_id_only` | position_local | 0.0000 | 0.0000 | 1.0000 | 10.0 | 0.1000 |
| `audio_envelope` | global | 0.0096 | 0.0909 | 0.1770 | 28.0 | 0.0737 |
| `audio_envelope` | position_local | 0.0144 | 0.2297 | 0.4737 | 11.0 | 0.1516 |

Duration and padding are constant by construction. The subject-only control uses the known PL subject-to-speaker cohort mapping. The audio-envelope control is an untrained correlation between EEG global field power and the candidate audio RMS envelope.

Sentence position is a strong shortcut in this candidate set. Formal model evaluation must report a fixed-size position-local pool in addition to the global pool.
