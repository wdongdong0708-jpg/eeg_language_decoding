# PL speech shortcut baselines

- Partition: `test`
- Queries: 295
- Unique candidates: 76
- Tie policy: pessimistic

| baseline | pool | R@1 | R@5 | R@10 | median rank | MRR |
|---|---|---:|---:|---:|---:|---:|
| `random` | global | 0.0136 | 0.0780 | 0.1559 | 36.0 | 0.0656 |
| `random` | position_local | 0.0475 | 0.2949 | 0.5254 | 10.0 | 0.1842 |
| `duration_only` | global | 0.0000 | 0.0000 | 0.0000 | 76.0 | 0.0132 |
| `duration_only` | position_local | 0.0000 | 0.0000 | 0.0000 | 20.0 | 0.0500 |
| `padding_mask_only` | global | 0.0000 | 0.0000 | 0.0000 | 76.0 | 0.0132 |
| `padding_mask_only` | position_local | 0.0000 | 0.0000 | 0.0000 | 20.0 | 0.0500 |
| `character_count_only` | global | 0.0000 | 0.0814 | 0.4271 | 12.0 | 0.1059 |
| `character_count_only` | position_local | 0.0000 | 0.6780 | 1.0000 | 4.0 | 0.2788 |
| `sentence_position_only` | global | 0.0339 | 0.9220 | 1.0000 | 2.0 | 0.4528 |
| `sentence_position_only` | position_local | 0.0339 | 0.9220 | 1.0000 | 2.0 | 0.4528 |
| `subject_id_only` | global | 0.0000 | 0.0000 | 0.0000 | 38.0 | 0.0263 |
| `subject_id_only` | position_local | 0.0000 | 0.0000 | 0.9864 | 10.0 | 0.1000 |
| `audio_envelope` | global | 0.0102 | 0.0644 | 0.1220 | 36.0 | 0.0618 |
| `audio_envelope` | position_local | 0.0373 | 0.2746 | 0.5186 | 10.0 | 0.1753 |

Duration and padding are constant by construction. The subject-only control uses the known PL subject-to-speaker cohort mapping. The audio-envelope control is an untrained correlation between EEG global field power and the candidate audio RMS envelope.

Sentence position is a strong shortcut in this candidate set. Formal model evaluation must report a fixed-size position-local pool in addition to the global pool.
