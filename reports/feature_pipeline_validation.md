# Feature pipeline validation

Validation date: 2026-07-30

## Passed

- `python -m pytest -q`: 36 tests passed.
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

## Strict-offline GPU inference

The official BERT and wav2vec model directories are already present in the local
Hugging Face cache, and `nvidia-smi` detects the RTX 5060 Ti 16 GB. The original
timeout was localized to a Transformers/PEFT adapter-config HTTP probe, not to a
PyTorch DLL import. The system proxy accepted the connection but timed out during
the Hugging Face TLS handshake.

Both extractors now make `local_files_only=True` strict: they enable Hub offline
mode, resolve the cached snapshot first, and give Transformers a local directory.
With all offline environment variables explicitly cleared before each process:

- BERT loaded from snapshot
  `8f23c25b06e129b6c986331a13d8d025a92cf0ea` and completed a CUDA sentence
  extraction in 5.751 s, producing sentence `[768]`, token `[11, 768]`, and
  character `[9, 768]` states.
- wav2vec loaded from snapshot
  `3155938c549b23eee16b1d4b55dcb161b7fe4bcf` and completed a one-second CUDA
  extraction in 6.050 s, producing 49 frames × 1024 features with support from
  0.000–0.985 s.

No model was downloaded and no formal feature artifact was written by the smoke
tests. End-to-end local-only GPU inference is now validated.
