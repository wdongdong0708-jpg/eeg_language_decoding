# Subject-layer v2: frozen BrainMagick-style test evaluation

## Protocol

- Seed: 42
- Test queries: 1,738
- Full unique test candidates: 438
- Subjects: 8
- Checkpoint selection: minimum validation loss only
- Best epoch: 4 (zero based)
- Best validation loss: 3.8878517602
- Test used for selection: no
- Subject layer: one bias-free 270 x 270 matrix per subject, after the
  initial 1x1 projection and before the temporal convolution stack

## Test metrics

| Aggregation | R@1 | R@5 | R@10 | Median rank | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| All-query micro | 3.57% | 12.03% | 19.10% | 45.00 | 0.0928 |
| Per-subject macro | 3.54% | 11.98% | 19.07% | 46.13 | 0.0924 |
| Analytical random | 0.23% | 1.14% | 2.28% | 219.50 | 0.0152 |

## Same-seed comparison with frozen no-subject v1

| Model | R@1 | R@5 | R@10 | Median rank | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| No subject layer, seed 42 | 3.11% | 11.28% | 17.43% | 56.00 | 0.0853 |
| Subject layer, seed 42 | 3.57% | 12.03% | 19.10% | 45.00 | 0.0928 |
| Difference | +0.46 pp | +0.75 pp | +1.67 pp | -11.00 | +0.0075 |

## Per-subject comparison

Every row uses the same global vocabulary of 438 candidates.

| Subject | Queries | v1 R@1 | v2 R@1 | Delta | v1 R@10 | v2 R@10 | Delta | v1 median | v2 median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 207 | 3.38% | 3.86% | +0.48 pp | 22.22% | 20.77% | -1.45 pp | 51.0 | 40.0 |
| 02 | 207 | 2.42% | 0.97% | -1.45 pp | 14.01% | 12.56% | -1.45 pp | 59.0 | 55.0 |
| 03 | 207 | 0.97% | 3.86% | +2.90 pp | 18.84% | 18.84% | +0.00 pp | 48.0 | 45.0 |
| 04 | 204 | 3.92% | 1.47% | -2.45 pp | 12.75% | 18.14% | +5.39 pp | 54.0 | 45.5 |
| 05 | 221 | 7.24% | 8.60% | +1.36 pp | 30.32% | 28.51% | -1.81 pp | 43.0 | 25.0 |
| 06 | 231 | 3.03% | 2.16% | -0.87 pp | 16.45% | 16.02% | -0.43 pp | 63.0 | 41.0 |
| 07 | 230 | 1.30% | 2.61% | +1.30 pp | 8.26% | 13.04% | +4.78 pp | 77.0 | 74.5 |
| 08 | 231 | 2.60% | 4.76% | +2.16 pp | 16.88% | 24.68% | +7.79 pp | 52.0 | 43.0 |

The same-seed result supports a positive aggregate effect, but it is not a
three-seed estimate: R@10 improves strongly for subjects 04, 07, and 08,
is unchanged for 03, and decreases for the other four subjects.
