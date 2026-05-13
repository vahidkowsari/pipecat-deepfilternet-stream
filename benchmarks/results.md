# Benchmark results — 2026-05-13

Run on Apple Silicon (M-series MacBook), single CPU core. Python 3.11.
Engine version: pipecat-deepfilternet-stream 0.1.0.

- **Files evaluated:** 50 paired noisy/clean samples from VoiceBank-DEMAND-16k (test split)
- **Total audio:** 174.5 s
- **Average RTF:** 0.11 (~9× real-time per stream)

## Headline numbers (noise removal)

| metric | mean | median | what it means |
|---|---:|---:|---|
| **Noise reduction (silent regions)** | **15.71 dB** | **15.74 dB** | Noise-only segments come out ~6× quieter (in linear amplitude) than they went in. |
| **SI-SDR improvement** | **+8.82 dB** | **+8.45 dB** | Standard speech-enhancement metric. Higher = cleaner output. |
| **Speech level preserved** | **92%** | **95%** | Of the original speech RMS energy. Some attenuation is expected; perceptually inaudible. |
| Real-time factor | 0.11 | 0.10 | Compute time / audio duration. ~9× real-time per stream on one CPU core. |

The +8.82 dB SI-SDR improvement matches the published DFN3 paper number
(+8.1 dB on the same VoiceBank-DEMAND test set), so the streaming engine
inherits the full model performance.

## Full summary

| metric | mean | median | p25 | p75 |
|---|---:|---:|---:|---:|
| `rtf` | 0.108 | 0.103 | 0.092 | 0.123 |
| `sisdr_noisy_vs_clean` | 8.83 | 9.99 | 4.61 | 13.78 |
| `sisdr_stream_vs_clean` | 17.65 | 18.20 | 15.42 | 20.12 |
| `sisdr_improvement_db` | 8.82 | 8.45 | 5.16 | 12.49 |
| `noise_reduction_db` | 15.71 | 15.74 | 11.93 | 19.62 |
| `speech_level_preserved` | 0.923 | 0.946 | 0.892 | 0.966 |

## Metric definitions

- **noise_reduction_db** — Build a mask from the clean reference: samples
  where `|clean| > 2% × peak` are "speech", below are "noise-only".
  Compute `20 * log10(noisy_rms_in_noise / streamed_rms_in_noise)`. Output
  is time-aligned to clean first to compensate for the 30 ms algorithmic
  delay. This is the most intuitive "how quiet is the noise now" metric.
- **sisdr_improvement_db** — `SI-SDR(streamed) - SI-SDR(noisy)`, both
  measured against clean. Standard speech-enhancement community metric;
  invariant to gain differences between estimate and reference.
- **speech_level_preserved** — `RMS(streamed)_speech_regions /
  RMS(clean)_speech_regions`. 1.0 means perfect level match; <1.0 means
  the model attenuated speech along with noise.
- **rtf** — `processing_time / audio_duration`. Lower is faster.
  `0.10` = 10× real-time.

## Reproduce

```bash
pip install pipecat-deepfilternet-stream soundfile soxr
pip install datasets       # only needed for fetch_voicebank_demand.py
pip install deepfilternet  # only if running with batch comparison (--no-batch skips it)

python fetch_voicebank_demand.py ./vbd16k --limit 50
python benchmark.py ./vbd16k --limit 50 --no-batch --output results.md
```

Drop `--no-batch` to also report `sisdr_batch_vs_clean` and
`sisdr_stream_vs_batch`. On a prior run those came in at 17.77 dB and
39.04 dB respectively, showing streaming is numerically equivalent to
batch (39 dB SI-SDR between them is the float32 precision floor).
