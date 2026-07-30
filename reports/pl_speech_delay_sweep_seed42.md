# PL EEG-delay sweep

- Common-support windows: 1,217
- Selection partition: validation
- Test model evaluation: not run
- Provisional selected delay: 500 ms

| delay | best epoch | validation loss | global R@1 | position-local R@1 |
|---:|---:|---:|---:|---:|
| 0 ms | 1 | 1.462020 | 0.0294 | 0.0294 |
| 100 ms | 0 | 1.525744 | 0.0196 | 0.0392 |
| 200 ms | 0 | 1.610091 | 0.0490 | 0.0490 |
| 300 ms | 1 | 1.548745 | 0.0392 | 0.0588 |
| 400 ms | 0 | 1.483976 | 0.0588 | 0.0686 |
| 500 ms | 0 | 1.458269 | 0.0392 | 0.0686 |

This is a single-seed engineering sweep, not a final scientific result. The selected delay must be confirmed with a predeclared schedule before one-time test evaluation.
