# pipecat-deepfilternet-stream

Per-frame streaming **DeepFilterNet3** (DFN3) noise suppression for [Pipecat](https://github.com/pipecat-ai/pipecat).

One hop in (10 ms @ 48 kHz), one hop out. **Batch-equivalent quality** at **~50 ms total filter delay** — vs ~170 ms for the chunked-batch streaming approach you'll find in most DFN3 wrappers, and vs the proprietary alternatives (Krisp, Koala, AIC) that require paid licenses.

This is a Python port of the per-frame inference loop in [Rikorose's `libDF/src/tract.rs`](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/tract.rs), built on Sonos' [`tract`](https://github.com/sonos/tract) Python bindings.

## Why this exists

Pipecat ships four noise filters out of the box:

| Filter | Free? | Quality | Streaming? |
|---|---|---|---|
| `RNNoiseFilter` | ✅ BSD | OK on stationary noise | per-frame |
| `KrispVivaFilter` | ❌ paid | best at background-voice removal | per-frame |
| `KoalaFilter` | ⚠️ free tier with AccessKey | good | per-frame |
| `AICFilter` | ❌ paid | bundles AEC | per-frame |
| **`DeepFilterNetStreamFilter`** (this package) | ✅ Apache-2.0 | best free option, comparable to Krisp on noise | **per-frame, ~50 ms delay** |

The publicly-available DFN3 Python integrations (including the one in [pipecat-ai/pipecat-extensions](https://github.com/pipecat-ai)) load DFN3's PyTorch model and feed it 100–200 ms chunks at a time, accepting the corresponding chunk-of-latency. This package skips PyTorch entirely at inference time and runs the bundled ONNX submodels through tract's pulsed-model transformation — the same trick the DFN3 Rust binary uses to do true per-frame streaming.

## Install

```bash
pip install pipecat-deepfilternet-stream
```

Wheels are available for Python 3.11–3.13 on Linux x86_64 and macOS (Intel + Apple Silicon). `tract` and `libdf` ship as pre-built wheels; no Rust toolchain required on the install side.

## Usage

Drop-in Pipecat filter:

```python
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat_deepfilternet_stream import DeepFilterNetStreamFilter

transport = FastAPIWebsocketTransport(
    websocket=ws,
    params=FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_filter=DeepFilterNetStreamFilter(),
        audio_out_enabled=True,
    ),
)
```

That's it. The filter resamples between the transport's sample rate and DFN3's native 48 kHz internally via SOXR streaming resamplers, so it works with 8 kHz Twilio/Vonage μ-law, 16 kHz pipelines, and 48 kHz web SDK audio.

For pre-warming the ONNX runnables in your FastAPI lifespan:

```python
from contextlib import asynccontextmanager
from pipecat_deepfilternet_stream import ensure_runnables

@asynccontextmanager
async def lifespan(app):
    ensure_runnables()  # pulses + optimises ONNX once, ~1 s on first call
    yield
```

### Lower-level API

If you're not using Pipecat:

```python
import numpy as np
from pipecat_deepfilternet_stream import PerFrameDfn, HOP_SIZE  # HOP_SIZE = 480

engine = PerFrameDfn()  # one per audio stream
for hop in audio_48k.reshape(-1, HOP_SIZE):  # 10 ms hops at 48 kHz
    clean_hop = engine.process_hop(hop)
```

`process_hop` is synchronous and returns the enhanced 480-sample hop. Algorithmic delay: 4 frames (40 ms) of model lookahead + 1 STFT pre-roll frame (10 ms). The first ~3 output hops correspond to the zero-initialised ring-buffer contents; subsequent output is valid.

### Examples

End-to-end scripts under [`examples/`](./examples):

- [`denoise_wav.py`](./examples/denoise_wav.py) — denoise a single file (any sample rate; SOXR resamples in/out).
- [`process_directory.py`](./examples/process_directory.py) — batch A/B over a folder, useful for demos.

```bash
python examples/denoise_wav.py noisy.wav clean.wav
```

## Performance

Measured on an Apple Silicon (M-series) MacBook, single CPU core:

| Metric | This package | Chunked DFN3 (160 ms chunks) | Notes |
|---|---:|---:|---|
| Real-time factor (RTF) | **0.10–0.13** | 0.19 | ~8–10× real-time per stream |
| Filter delay | **30 ms** | 170 ms | model lookahead + STFT pre-roll |
| SI-SDR vs batch `enhance()` | **45 dB** | ~20 dB | float-precision identical |
| SI-SDR vs clean (VoiceBank-DEMAND) | matches batch ±0.2 dB | matches batch ±0.1 dB | within model performance |

A 32-concurrent host has plenty of CPU headroom per stream.

See [`benchmarks/results.md`](./benchmarks/results.md) for a full
reproducible benchmark on 50 paired noisy/clean files from
VoiceBank-DEMAND-16k.

## How it works

DFN3's ONNX bundle does not expose GRU hidden states or conv2d time-kernel buffers as graph I/O. Calling them with `S=1` frames through onnxruntime collapses model quality (we measured a ~-23 dB SI-SDR collapse versus the batch enhance path) because the recurrent and temporal-context state resets to zero each call.

Sonos' `tract` runtime has a feature called the **pulsed model transformation**. When applied with `pulse=1`, tract rewrites every stateful op (GRU, time-domain Conv2d, Scan, etc.) to maintain per-instance internal state across `state.run()` calls. The DFN3 Rust binary uses this transformation via the Rust API. tract's Python bindings expose the same transform via `model.pulse("S", 1)`.

There's also a second subtle bit: `libdf.DF.analysis(input, reset=False)` is required for streaming. The default `reset=None` silently resets the STFT OLA state between per-hop calls, making per-hop output drift away from batch output by ~3e-3 per frame. With `reset=False`, per-hop output is bit-exact identical to a batch call on the concatenated audio.

Combine those two and you get per-frame DFN3 inference in Python with batch quality.

## Limitations

- **No echo cancellation.** DFN3 is a noise suppressor; far-end TTS bleeding through near-end mic still gets through. Layer WebRTC APM in front of this filter if you need AEC.
- **Doesn't remove background voices.** Other humans speaking in the same room pass through. That's the differentiating feature of Krisp.
- **CPU only.** The ONNX submodels are small enough that adding a GPU path doesn't pay off; tract's CPU performance is already well into real-time.

## Development

```bash
git clone https://github.com/vahidkowsari/pipecat-deepfilternet-stream
cd pipecat-deepfilternet-stream
pip install -e ".[test]"
pytest
```

To run the reproducible benchmark against VoiceBank-DEMAND-16k:

```bash
pip install soundfile datasets deepfilternet  # batch reference path
python benchmarks/fetch_voicebank_demand.py ./vbd16k --limit 50
python benchmarks/benchmark.py ./vbd16k --limit 50 --output benchmarks/results.md
```

See [`CLAUDE.md`](./CLAUDE.md) for an orientation to the codebase
(invariants to preserve, per-stream vs process-wide state, where the
critical `reset=False` / `pulse(S, 1)` calls live).

## Contributing

Issues and PRs are welcome. Particularly interested in:

- Bug reports with reproducible audio (a 5–10 s WAV is ideal).
- Wheel-build issues on platforms not in the CI matrix.
- Cleaner alternatives to the manual ERB inverse filterbank in `engine.py`.

Please run `pytest` and, for engine changes, the benchmark — keep
`sisdr_stream_vs_batch` above 25 dB (it's the streaming-identity canary).

## License

Apache-2.0. The bundled ONNX models come from [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (also Apache-2.0).

## Acknowledgements

- [Hendrik Schröter](https://github.com/Rikorose) for DeepFilterNet3 and the original Rust `tract`-based per-frame implementation that this package ports to Python.
- [Sonos](https://github.com/sonos/tract) for `tract` and its pulsed-model transformation, without which per-frame DFN3 in Python would not be possible.
- The [Pipecat](https://github.com/pipecat-ai/pipecat) team for the `BaseAudioFilter` interface and SOXR streaming resampler.
