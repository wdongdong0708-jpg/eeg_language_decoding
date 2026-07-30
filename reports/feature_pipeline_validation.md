# Feature pipeline validation

Validation date: 2026-07-30

## Passed

- `python -m pytest -q`: 30 tests passed.
- `python -m compileall -q src scripts tests`: passed.
- All YAML files parse with `yaml.safe_load`.
- `git diff --check`: passed.
- Local XLSX reader verified the required Ch1 cells and preserved exact Excel
  row numbers.
- A one-second block from
  `materials&embeddings/audio/littleprince_f1/audio_0.wav` loaded as 12,000
  mono `float32` samples from the source 12 kHz stereo file.
- Text tests verify that batch padding is excluded from sentence pooling and
  that token/character offsets survive no-pickle NPZ round trips.
- Audio tests verify the standard wav2vec convolution geometry (320-sample
  stride, 400-sample receptive field), source-time frame coordinates, inherited
  split validation, and no-pickle NPZ round trips.

## Environment issue requiring follow-up

The official BERT and wav2vec model directories are already present in the local
Hugging Face cache, and `nvidia-smi` detects the RTX 5060 Ti 16 GB. However, two
minimal `bm5060` attempts to import PyTorch for real model inference did not
return within 120 seconds (one through `conda run`, one through the environment
Python directly, including a CPU-targeted attempt). The exact child processes
were terminated after the timeout; no model was downloaded and no feature
artifact was written.

Accordingly, this phase validates the extraction contracts, real workbook/audio
I/O, serialization, and boundary logic, but does **not** claim that end-to-end
GPU inference has passed. Before bulk extraction, diagnose the PyTorch DLL/import
stall and rerun a one-sentence/one-second local-only smoke test.
