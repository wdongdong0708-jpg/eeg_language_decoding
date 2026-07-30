# PL speech shortcut baselines

- Partition: `test`
- Queries: 150
- Unique candidates: 38
- Tie policy: pessimistic

| baseline | pool | R@1 | R@5 | R@10 | median rank | MRR |
|---|---|---:|---:|---:|---:|---:|
| `random` | global | 0.0133 | 0.0933 | 0.2067 | 22.0 | 0.0873 |
| `random` | position_local | 0.0200 | 0.2133 | 0.4333 | 12.0 | 0.1484 |
| `duration_only` | global | 0.0000 | 0.0000 | 0.0000 | 38.0 | 0.0263 |
| `duration_only` | position_local | 0.0000 | 0.0000 | 0.0000 | 20.0 | 0.0500 |
| `padding_mask_only` | global | 0.0000 | 0.0000 | 0.0000 | 38.0 | 0.0263 |
| `padding_mask_only` | position_local | 0.0000 | 0.0000 | 0.0000 | 20.0 | 0.0500 |
| `character_count_only` | global | 0.0000 | 0.1600 | 1.0000 | 8.0 | 0.1583 |
| `character_count_only` | position_local | 0.0000 | 0.6333 | 1.0000 | 4.0 | 0.2569 |
| `sentence_position_only` | global | 0.0333 | 1.0000 | 1.0000 | 2.0 | 0.4594 |
| `sentence_position_only` | position_local | 0.0333 | 1.0000 | 1.0000 | 2.0 | 0.4594 |
| `subject_id_only` | global | 0.0000 | 0.0000 | 0.0000 | 19.0 | 0.0526 |
| `subject_id_only` | position_local | 0.0000 | 0.0000 | 1.0000 | 10.0 | 0.1001 |
| `audio_envelope` | global | 0.0467 | 0.1467 | 0.2733 | 20.0 | 0.1275 |
| `audio_envelope` | position_local | 0.0667 | 0.2600 | 0.5000 | 10.5 | 0.1852 |

Duration and padding are constant by construction. The subject-only control uses the known PL subject-to-speaker cohort mapping. The audio-envelope control is an untrained correlation between EEG global field power and the candidate audio RMS envelope.

Sentence position is a strong shortcut in this candidate set. Formal model evaluation must report a fixed-size position-local pool in addition to the global pool.
