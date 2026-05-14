# A Free Noise Filter for Your Pipecat Voice Agent

*Krisp-quality noise suppression for phone and WebSocket voice agents — without the license fee, without the latency tax. A drop-in Pipecat `BaseAudioFilter` for streaming DeepFilterNet3.*

---

If you've built a voice agent on [Pipecat](https://github.com/pipecat-ai/pipecat) and run it over a phone line, you already know the problem. Twilio drops 8 kHz μ-law audio into your pipeline, the caller is in a coffee shop or driving with the window cracked, your STT gets noisy partial transcripts, your LLM responds to half-heard sentences, and the conversation falls apart.

Pipecat ships four noise filters out of the box. None of them is the obvious choice:

| Filter | Free? | Quality | Latency |
|---|---|---|---|
| `RNNoiseFilter` | ✅ BSD | OK on stationary noise | per-frame (~10 ms) |
| `KrispVivaFilter` | ❌ paid | best at background-voice removal | per-frame |
| `KoalaFilter` | ⚠️ free tier w/ AccessKey | good | per-frame |
| `AICFilter` | ❌ paid | bundles AEC | per-frame |

[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) — Apache-2.0, state-of-the-art on the DNS Challenge, comparable to Krisp on non-voice background noise — should have been the fifth row. It wasn't, because every public Python integration runs it in 100–200ms chunks. That's a hard latency floor of ~170ms on the filter alone. In a voice agent pipeline where you're already counting milliseconds against the time-to-first-response budget, adding 170ms of preprocessing before STT even sees the audio is a non-starter.

So Pipecat users end up either paying Krisp/AIC, accepting RNNoise's limitations, or coding around the problem. This article is about a fifth option I built and open-sourced: [`pipecat-deepfilternet-stream`](https://github.com/vahidkowsari/pipecat-deepfilternet-stream).

## The drop-in

```bash
pip install pipecat-deepfilternet-stream
```

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

That's it. The filter handles SOXR streaming resampling between the transport rate and DFN3's native 48 kHz internally, so it works with:

- **8 kHz μ-law** (Twilio Voice, Vonage, Plivo, SignalWire)
- **16 kHz PCM** (most server-side voice pipelines, Deepgram default)
- **24 kHz / 48 kHz** (browser WebRTC, web SDK audio)

Total filter delay: **~50 ms**, vs ~170 ms for the chunked DFN3 wrappers, vs per-frame for Krisp/Koala/AIC.

If you're running a FastAPI service, pre-warm the ONNX runnables once at boot so the first call doesn't pay the ~1s init cost:

```python
from contextlib import asynccontextmanager
from pipecat_deepfilternet_stream import ensure_runnables

@asynccontextmanager
async def lifespan(app):
    ensure_runnables()  # ~1 s, once per process
    yield
```

## What you actually get

I benchmarked it against the standard VoiceBank-DEMAND-16k test set — 50 paired noisy/clean files, the same benchmark the DFN3 paper reports against. Numbers on an M-series MacBook, single CPU core:

| Metric | This filter | Chunked DFN3 wrappers | What this means for your agent |
|---|---:|---:|---|
| Filter delay | **~50 ms** | ~170 ms | end-to-end response time |
| Real-time factor | **0.10–0.13** | 0.19 | a single host can serve 30+ concurrent calls per core |
| Noise reduction | **15.7 dB** | 15.7 dB | how much background noise is gone |
| SI-SDR improvement | **+8.8 dB** | +8.8 dB | the DFN3 paper's headline number |
| Speech level preserved | **92%** | 92% | the filter is not eating your caller's voice |
| Quality vs batch DFN3 | **45 dB SI-SDR identity** | ~20 dB | per-hop output ≡ batch within float precision |

The takeaway: **batch DFN3 quality at 1/3 the latency**, on a single CPU core, with no GPU, no paid AccessKey, no API call out of your pipeline.

In practice for a phone agent that means: STT transcripts come back cleaner on the same noisy calls, the LLM stops hallucinating around bad transcripts, and the user-perceived latency budget is the same as it was with RNNoise.

## A note on echo and background voices

Be honest with yourself about which problem you're solving. DFN3 — and therefore this filter — is a **noise suppressor**, not a full front-end:

- **No AEC.** If your far-end TTS is bleeding through the near-end microphone (always the case on phone bridges and most browser setups without echo cancellation), DFN3 will pass it through. The agent will hear itself talking and the LLM will get confused. Layer WebRTC APM in front of this filter, or use Pipecat's `AICFilter` (which bundles AEC) if budget allows.
- **No background-voice removal.** Other humans in the same room talking? Comes through. That's specifically the thing Krisp's voice-isolation model is trained to suppress, and it remains a real differentiator for them.

If you're on a clean phone bridge with hardware AEC at the carrier (e.g. SIP gateways) and your problem is environmental noise — fans, traffic, café chatter, wind, keyboard clack — this filter is the right call. If your problem is the agent hearing its own TTS, fix that first.

## What changed under the hood (the short version)

If you only care that it works, skip this section. If you've already tried wiring DFN3 into your Pipecat agent yourself and given up, this is for you.

The reason every public Python DFN3 integration runs in 100–200ms chunks is that DFN3's ONNX exports do not expose the GRU hidden states or the conv2d time-kernel buffers as graph I/O. Call them with `S=1` (one frame at a time) through `onnxruntime` and the recurrent state silently resets to zero on every call. I measured the quality collapse at ~23 dB SI-SDR — usable in batch, robot-gargling in streaming. So the Python wrappers gather chunks, accept the chunk-of-latency, and move on.

The Rust binary that ships with DFN3 does not have this problem. It uses Sonos' [`tract`](https://github.com/sonos/tract) runtime and applies its **pulsed-model transformation**, which rewrites every stateful op (GRU, time-Conv2d, Scan) so internal state persists across `state.run()` calls. The transformation is exposed in tract's Python bindings as `model.pulse("S", 1)` — same trick, same result, no Rust required at install time.

The other gotcha is `libdf.DF.analysis()`. The Python binding defaults `reset=None`, which silently resets the streaming STFT overlap-add state between per-hop calls. You have to pass `reset=False` explicitly — `None` is *not* "leave it alone." With both invariants — pulsed tract + `reset=False` — per-hop output is bit-exact identical to a batch `df.enhance.enhance()` call on the same audio.

That's the whole trick. The full per-frame inference loop is ~200 lines of Python and is a direct port of Rikorose's [`libDF/src/tract.rs`](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/tract.rs).

## What's in the dependency stack

Worth knowing what you're actually pulling in when you `pip install` this. Five non-trivial pieces:

### `deepfilternet` / `deepfilterlib` (the model)

[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) is Hendrik Schröter's 2023 speech enhancement model. It's an Apache-2.0 hybrid architecture: an ERB-band magnitude mask for the upper frequency range (perceptually-motivated, mirrors how the cochlea bins frequencies) plus a complex-valued **deep filter** (`df_op`) over the lower 96 frequency bins that handles phase. Together they preserve speech harmonics that pure magnitude-mask denoisers (RNNoise, classical Wiener) shred.

The Python distribution comes in two pieces:

- **`deepfilternet`** — the PyTorch reference implementation, training code, and the offline `df.enhance.enhance()` path used by every chunked Python wrapper.
- **`deepfilterlib`** — Python bindings to `libdf` (Rust), the C-callable library that handles the STFT, the ERB filterbank, the ERB feature extraction, and the streaming overlap-add state. This is what gets called per-hop in this package — we don't use the PyTorch path at inference time at all.

Three ONNX submodels (`enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`) are bundled in this package's wheel — these are the official exports from the Rikorose release. Total weight bundle is ~2 MB.

### `tract` (the inference runtime)

[Sonos `tract`](https://github.com/sonos/tract) is the Rust ONNX runtime that powers Sonos smart speakers. Two reasons it's the right runtime here, and `onnxruntime` is not:

1. **Pulsed-model transformation.** Already covered above — this is the single feature that makes per-frame DFN3 inference possible. `onnxruntime` has nothing equivalent.
2. **It's small and CPU-focused.** No CUDA, no protobuf-versioning surprises, ~5 MB wheel. For a per-hop CPU workload that has to fit alongside an STT model and an LLM client in the same Python process, `tract`'s footprint is a feature.

The Python bindings (`tract` on PyPI) landed around 0.21 and are still relatively obscure — most of the Python ONNX ecosystem reaches for `onnxruntime` reflexively. For streaming RNN/conv-state models, that reflex is wrong.

### `pipecat-ai` (the framework)

[Pipecat](https://github.com/pipecat-ai/pipecat) provides the `BaseAudioFilter` interface this package implements and the SOXR streaming resampler we plug into. The filter contract is small — `start(sample_rate)`, `filter(bytes) -> bytes`, `stop()` — which is the whole reason it was possible to write this as a 100-line wrapper around the engine.

Pipecat also handles the audio-frame transport plumbing (Twilio, Daily, FastAPI WebSocket, Vonage, etc.) so this filter doesn't have to know whether it's sitting behind a SIP gateway or a browser. Same code path for every transport.

### `soxr` (resampling)

DFN3 runs at 48 kHz natively. Your transport is almost certainly not 48 kHz — 8 kHz μ-law from Twilio, 16 kHz from most server pipelines, 24/48 kHz from browsers. SOXR (via Pipecat's `SOXRStreamAudioResampler`) handles both directions with stateful resamplers that don't reset between calls.

This package defaults to `quality="QQ"` (Quick — same as Pipecat's RNNoise filter) for minimum latency. "HQ" is available if you want better resample fidelity at a small latency cost; for voice-agent use cases the difference is inaudible and QQ saves a measurable chunk of per-hop CPU.

### NumPy

The glue. The engine uses NumPy for the ERB inverse filterbank multiply, the feature-normalization EMA state, the complex-valued df_op time-window multiply, and the ring buffers between hops. ~30 lines of actual NumPy work per hop. No SciPy, no Numba, no Cython.

### The ONNX models themselves

A note worth making since "ONNX" can mean a lot of things: these are *the* official ONNX exports from the upstream DeepFilterNet3 release. Not a re-export, not a quantized variant, not a community fork. Same weights as the Rust binary. Same weights as the PyTorch `df.enhance.enhance()` reference path. That's why the streaming output matches batch to 45 dB SI-SDR — we are literally running the same numbers, just one frame at a time with the state machine made explicit by `tract`.

## When to use which Pipecat filter

To save you a decision tree:

- **Stationary background noise only, latency-critical, free?** RNNoise is fine.
- **Environmental noise (fans, traffic, café, wind), free, low-latency?** This filter.
- **Other humans speaking nearby need to be muted?** Krisp. There's no free equivalent for this and DFN3 explicitly doesn't attempt it.
- **You need AEC bundled?** AIC, or layer WebRTC APM in front of this filter.
- **You want a free-tier with quality close to Krisp on non-voice noise?** Koala or this filter; pick based on whether you want an AccessKey dependency.

## Origin story

I built this for a production voice-agent system because none of the existing options fit. RNNoise wasn't enough on the noise floor we were seeing, the chunked DFN3 wrappers added 120ms we couldn't afford, and a paid Krisp/AIC line item didn't make sense for a feature where we already had a high-quality open-source model available.

The integration is small. The hard part wasn't the code — it was finding out that `tract.pulse("S", 1)` existed in the Python bindings and that `libdf`'s `reset=None` argument was a footgun. Both pieces of information are buried in the Rust source of upstream DFN3, and as far as I can tell nobody had wired them up in Python yet.

So: open-sourced, Apache-2.0, matching upstream. The hope is that the next Pipecat user trying to use DFN3 doesn't have to re-derive any of this from Rikorose's Rust source the way I did.

## Repository and references

**[github.com/vahidkowsari/pipecat-deepfilternet-stream](https://github.com/vahidkowsari/pipecat-deepfilternet-stream)** — install, examples, reproducible benchmark.

References and acknowledgements:

- **DeepFilterNet3 paper**: Hendrik Schröter et al., *"DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement"*, Interspeech 2023. [arXiv:2305.08227](https://arxiv.org/abs/2305.08227).
- **DeepFilterNet repository**: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — the upstream Rust + PyTorch implementation, ONNX exports, and the `libDF/src/tract.rs` reference loop this package ports to Python.
- **tract**: [sonos/tract](https://github.com/sonos/tract) — the runtime whose pulsed-model transformation makes streaming inference of stateful ONNX models possible from Python.
- **Pipecat**: [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) — the real-time voice-agent framework this filter targets.
- **VoiceBank-DEMAND-16k**: [JacobLinCool/VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) — the noise-suppression benchmark used in the committed results.

Issues and PRs welcome. If you ship this in a production voice agent, I'd genuinely like to hear how it performs on real call audio — that's the feedback loop that matters.
