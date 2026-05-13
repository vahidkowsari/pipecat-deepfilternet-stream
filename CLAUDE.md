# CLAUDE.md

Guidance for Claude Code (and similar agents) working in this repository.

## What this package is

A Python port of the per-frame inference loop from Rikorose's
`libDF/src/tract.rs`, packaged as a Pipecat `BaseAudioFilter`. It runs
DeepFilterNet3 with **one hop in, one hop out** (10 ms @ 48 kHz, ~50 ms
total filter delay) instead of the 100–200 ms chunked-batch approach used
by every other public Python DFN3 integration.

Two non-obvious ingredients make this work — preserve them when refactoring:

1. **`tract.pulse("S", 1)`** on each ONNX submodel. Sonos' tract rewrites
   stateful ops (GRU, time-domain Conv2d, Scan) so internal state persists
   across `state.run()` calls. `onnxruntime` cannot do this — calling DFN3
   ONNX with `S=1` through onnxruntime collapses quality by ~23 dB SI-SDR.
2. **`libdf.DF.analysis(..., reset=False)`** and `.synthesis(..., reset=False)`.
   The default `reset=None` silently resets the streaming STFT OLA state
   between per-hop calls. `reset=False` keeps OLA continuous; this is what
   makes per-hop output bit-exact identical to a batch `enhance()` call.

If either invariant is broken, the engine still runs but quality regresses
silently. The benchmark's `sisdr_stream_vs_batch` metric is the canary —
expect >25 dB; <10 dB means streaming has diverged from batch.

## Layout

```
src/pipecat_deepfilternet_stream/
  engine.py     # PerFrameDfn — the actual per-hop inference loop. Pure-Python port of libDF/src/tract.rs.
  filter.py     # DeepFilterNetStreamFilter — Pipecat BaseAudioFilter wrapping engine + SOXR resampling.
  onnx_models/  # enc.onnx, erb_dec.onnx, df_dec.onnx (bundled from Rikorose/DeepFilterNet, Apache-2.0).
tests/          # smoke tests; deep numerical validation lives in benchmarks/.
benchmarks/     # VoiceBank-DEMAND-16k harness (SI-SDR + RTF). results.md is committed.
examples/       # denoise_wav.py (single file), process_directory.py (batch A/B).
```

The ONNX files are tracked in git and shipped in the wheel via
`pyproject.toml`'s `force-include`. Don't add them to `.gitignore`.

## Constants you'll see across the code

| Symbol | Value | Source |
|---|---|---|
| `DFN_SAMPLE_RATE` | 48000 | DFN3 native rate |
| `HOP_SIZE` | 480 | 10 ms @ 48 kHz |
| `FFT_SIZE` | 960 | 50% overlap |
| `N_FREQS` | 481 | FFT_SIZE/2 + 1 |
| `NB_DF` | 96 | DF (deep filter) bins |
| `NB_ERB` | 32 | ERB bands |
| `DF_ORDER` | 5 | df_op time-window length |
| `DF_LOOKAHEAD` / `CONV_LOOKAHEAD` | 2 | model lookahead frames |
| `ALPHA` | 0.99 | feature-norm EMA |
| `MEAN_NORM_INIT`, `UNIT_NORM_INIT` | from `libDF/src/lib.rs` | feature-norm initial state |

These are model-bound — don't change them unless re-exporting the ONNX models.

## Per-stream vs process-wide state

- **Process-wide:** the three pulsed tract runnables (`_runnables` dict in
  `engine.py`). Created once by `ensure_runnables()`, idempotent.
- **Per-stream:** everything in `PerFrameDfn.__init__` — tract `spawn_state()`,
  the `libdf.DF` instance, feature-norm EMA arrays, and the two `deque`
  ring buffers. One `PerFrameDfn` per audio stream.

If you add new state, classify it correctly. Sharing per-stream state
across calls cross-talks audio between sessions.

## Development

```bash
pip install -e ".[test]"
pytest                          # smoke tests
cd benchmarks
python fetch_voicebank_demand.py ./vbd16k --limit 50
python benchmark.py ./vbd16k --limit 50 --output results.md
```

Benchmarks need the heavyweight `deepfilternet` (PyTorch) package for the
batch reference path; pass `--no-batch` to skip it.

## When making changes

- Edits to `engine.py`: rerun the benchmark and confirm `sisdr_stream_vs_batch`
  stays above 25 dB. That's the contract — quality identity vs batch.
- Edits to `filter.py`: changes to resampling, buffering, or hop draining
  should be exercised against at least one non-48 kHz transport rate
  (16 kHz is the common case; 8 kHz μ-law is the edge case).
- New code paths near `analysis()` / `synthesis()`: double-check
  `reset=False` is passed. The default `reset=None` is a footgun.
- Don't add an `onnxruntime` path "as a fallback". onnxruntime cannot
  maintain hidden state between `S=1` calls; it will silently produce
  ~20 dB worse output than the pulsed-tract path. If tract import fails
  the right behavior is to surface the error.

## House style

- Keep comments load-bearing: explain *why* (cite `libDF/src/lib.rs`
  line, or which DFN3 paper section a constant comes from). Avoid
  restating what the next line of code obviously does.
- No backwards-compat shims for unreleased internal API. This is a
  fresh package; rename freely.
- Public surface is the `__all__` in `__init__.py`. If you add an entry,
  also document it in the README's "Lower-level API" section.
