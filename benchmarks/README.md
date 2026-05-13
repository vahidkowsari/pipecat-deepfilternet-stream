# Benchmarks

Reproducible benchmark for the per-frame DFN3 streaming engine. Validates
two claims:

1. **Streaming output is bit-equivalent to batch.** Per-frame `process_hop`
   matches `df.enhance.enhance()` within float precision (SI-SDR >25 dB
   between the two paths).
2. **Quality matches batch DFN3.** SI-SDR against the clean reference is
   within a fraction of a dB of the offline ceiling.

## Quick start

```bash
pip install pipecat-deepfilternet-stream
pip install soundfile datasets deepfilternet  # for benchmark deps + ground truth
python fetch_voicebank_demand.py ./vbd16k --limit 50
python benchmark.py ./vbd16k --limit 50 --output results.md
```

## Dataset

[VoiceBank-DEMAND-16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k)
is the canonical DFN3 benchmark — paired noisy/clean speech recordings at
16 kHz. The standard test set has 824 files; the script defaults to a
50-file subset for speed.

Resampling: the engine processes at 48 kHz internally. Inputs are
upsampled via SOXR ("HQ" quality in the benchmark, "QQ" in the production
filter for lower latency).

## Metrics

| metric | meaning |
|---|---|
| `rtf` | real-time factor (compute time / audio duration). Lower is faster. |
| `sisdr_noisy_vs_clean` | baseline: how dirty the input is. |
| `sisdr_stream_vs_clean` | streamed output vs ground truth. Higher is better. |
| `sisdr_batch_vs_clean` | reference batch `enhance()` vs ground truth. The model ceiling. |
| `sisdr_stream_vs_batch` | identity validation. >25 dB means streaming ≡ batch numerically. |

SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) is the standard
metric for speech enhancement benchmarks: it's invariant to gain
differences between estimate and reference, so a perfectly attenuated
signal still scores high.

## Results

See [`results.md`](./results.md) for the latest committed run.

## Reproducing the published results

Hardware: Apple Silicon (M-series), single CPU core. Disable batch
verification with `--no-batch` to skip the PyTorch path (much faster).
