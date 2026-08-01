# Frozen BrainMagick-style three-seed test evaluation

- Seeds: 42, 43, 44
- Test queries: 1738
- Full test candidates: 438
- Subjects: 8
- Dispersion: sample standard deviation across seeds (ddof=1)

| Aggregation | R@1 | R@5 | R@10 | Median rank | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BrainMagick-compatible micro | 3.53% ± 0.59% | 10.91% ± 0.89% | 17.09% ± 0.32% | 54.67 ± 1.53 | 0.0872 ± 0.0050 |
| Per-subject macro | 3.55% ± 0.59% | 10.99% ± 0.88% | 17.17% ± 0.29% | 56.42 ± 1.22 | 0.0877 ± 0.0051 |
| Analytical random | 0.23% | 1.14% | 2.28% | 219.50 | 0.0152 |

The primary row follows BrainMagick's published aggregation: all test queries against the same full unique test-segment vocabulary, followed by mean/sample-standard-deviation across three model seeds. The per-subject macro row is an additional diagnostic.
