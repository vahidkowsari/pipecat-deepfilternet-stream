"""Benchmark the per-frame engine over a directory of paired noisy/clean wavs.

Reports for each file:
  - RTF (real-time factor)
  - SI-SDR (streamed output vs clean reference)
  - SI-SDR (streamed vs batch enhance — validates streaming ≡ batch)
  - SI-SDR (batch vs clean — DFN3 absolute ceiling)

Aggregates: mean / median / p25 / p75 across the dataset.

Expects a directory layout like:

    samples_dir/
    ├── noisy/
    │   ├── p232_005.wav
    │   └── ...
    └── clean/
        ├── p232_005.wav
        └── ...

The standard benchmark dataset is VoiceBank-DEMAND-16k. Get it via:

    pip install datasets
    python -c "from datasets import load_dataset; \\
               load_dataset('JacobLinCool/VoiceBank-DEMAND-16k', split='test')"

Or use the script in fetch_voicebank_demand.py.

Usage:
    python benchmark.py /path/to/vbd16k --limit 50 --output results.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def align(ref: np.ndarray, est: np.ndarray, max_lag: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    a, b, _ = _align_with_lag(ref, est, max_lag)
    return a, b


def _align_with_lag(ref: np.ndarray, est: np.ndarray, max_lag: int = 4000) -> tuple[np.ndarray, np.ndarray, int]:
    n = min(len(ref), len(est))
    a, b = ref[:n], est[:n]
    c = np.correlate(b, a, mode="full")
    center = n - 1
    window = c[max(0, center - max_lag) : center + max_lag + 1]
    lag = int(window.argmax()) - min(max_lag, center)
    if lag > 0:
        b = b[lag:]; a = a[: len(b)]
    elif lag < 0:
        a = a[-lag:]; b = b[: len(a)]
    m = min(len(a), len(b))
    return a[:m], b[:m], lag


def stream_engine(audio_48k: np.ndarray) -> tuple[np.ndarray, float]:
    """Process audio through the per-frame engine. Returns (output_48k, rtf)."""
    from pipecat_deepfilternet_stream import HOP_SIZE, PerFrameDfn

    engine = PerFrameDfn()
    n = (audio_48k.size // HOP_SIZE) * HOP_SIZE
    audio_48k = audio_48k[:n]
    t0 = time.perf_counter()
    chunks = [engine.process_hop(audio_48k[i : i + HOP_SIZE]) for i in range(0, n, HOP_SIZE)]
    elapsed = time.perf_counter() - t0
    rtf = elapsed / (n / 48000)
    return np.concatenate(chunks), rtf


_batch_model = None
_batch_df_state = None


def batch_enhance(audio_48k: np.ndarray) -> np.ndarray:
    """Reference batch enhance via df.enhance.enhance().

    Loads the DFN3 PyTorch model and df_state once, caches them module-wide.
    df.enhance.enhance() resets the model's GRU hidden states internally each
    call, so the cached state is safe to reuse across files.
    """
    global _batch_model, _batch_df_state
    import torch
    from df.enhance import enhance, init_df

    if _batch_model is None:
        _batch_model, _batch_df_state, _ = init_df()
        _batch_model.eval()
    audio_t = torch.from_numpy(audio_48k).unsqueeze(0)
    return enhance(_batch_model, _batch_df_state, audio_t, pad=True).squeeze(0).numpy()


def maybe_resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    import soxr
    return soxr.resample(audio, src, dst, quality="HQ").astype(np.float32)


def collect_pairs(root: Path, limit: int) -> list[tuple[Path, Path]]:
    noisy_dir = root / "noisy"
    clean_dir = root / "clean"
    if not (noisy_dir.is_dir() and clean_dir.is_dir()):
        raise SystemExit(f"Expected {root}/noisy and {root}/clean directories.")
    pairs = []
    for n in sorted(noisy_dir.glob("*.wav")):
        c = clean_dir / n.name
        if c.exists():
            pairs.append((n, c))
        if len(pairs) >= limit:
            break
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark pipecat-deepfilternet-stream.")
    p.add_argument("samples_dir", type=Path, help="Dir with noisy/ and clean/ subdirs")
    p.add_argument("--limit", type=int, default=50, help="Max paired files to evaluate")
    p.add_argument("--no-batch", action="store_true",
                   help="Skip batch enhance comparison (no df.enhance dep needed)")
    p.add_argument("--output", type=Path, default=None,
                   help="Write a markdown report to this path")
    args = p.parse_args()

    import soundfile as sf
    pairs = collect_pairs(args.samples_dir, args.limit)
    if not pairs:
        raise SystemExit("No paired noisy/clean wavs found.")
    print(f"Evaluating {len(pairs)} paired files from {args.samples_dir}")

    rows: list[dict] = []
    total_dur = 0.0
    total_rt = 0.0
    for noisy_p, clean_p in pairs:
        noisy, sr_n = sf.read(str(noisy_p), dtype="float32", always_2d=True)
        clean, sr_c = sf.read(str(clean_p), dtype="float32", always_2d=True)
        if sr_n != sr_c:
            print(f"  skipping {noisy_p.name}: mismatched sample rates")
            continue
        noisy = noisy.mean(axis=1)
        clean = clean.mean(axis=1)

        noisy_48k = maybe_resample(noisy, sr_n, 48000)
        clean_48k = maybe_resample(clean, sr_c, 48000)

        out_48k, rtf = stream_engine(noisy_48k)
        # The streamed output has ~30 ms of algorithmic delay (model lookahead
        # + STFT pre-roll). Recover the integer lag and re-time-align the
        # output to clean/noisy before computing time-domain noise metrics.
        c_a, y_a, lag = _align_with_lag(clean_48k, out_48k)
        sisdr_clean = si_sdr(c_a, y_a)
        n_a, c_b = align(clean_48k, noisy_48k)
        sisdr_noisy = si_sdr(n_a, c_b)

        # Shift output to be aligned with clean for the masked-RMS metrics.
        out_aligned = out_48k[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, dtype=np.float32), out_48k]) if lag < 0 else out_48k
        n = min(len(noisy_48k), len(out_aligned), len(clean_48k))
        clean_n = clean_48k[:n]
        noisy_n = noisy_48k[:n]
        out_n = out_aligned[:n]

        # Speech-vs-noise mask from clean: |clean| > 2% of peak is "speech",
        # below is noise-only. Using clean (not noisy) avoids confusion from
        # the noise itself triggering the threshold.
        c_env = np.abs(clean_n)
        speech_mask = c_env > c_env.max() * 0.02
        noise_mask = ~speech_mask
        if noise_mask.any() and noise_mask.sum() > 100:
            noisy_noise_rms = float(np.sqrt((noisy_n[noise_mask] ** 2).mean()) + 1e-12)
            out_noise_rms = float(np.sqrt((out_n[noise_mask] ** 2).mean()) + 1e-12)
            nr_db = 20 * np.log10(noisy_noise_rms / out_noise_rms)
        else:
            nr_db = float("nan")
        if speech_mask.any():
            clean_sp_rms = float(np.sqrt((clean_n[speech_mask] ** 2).mean()) + 1e-12)
            out_sp_rms = float(np.sqrt((out_n[speech_mask] ** 2).mean()) + 1e-12)
            speech_preserved = out_sp_rms / clean_sp_rms
        else:
            speech_preserved = float("nan")

        row = {
            "file": noisy_p.name,
            "dur_s": len(noisy) / sr_n,
            "rtf": rtf,
            "sisdr_noisy_vs_clean": sisdr_noisy,
            "sisdr_stream_vs_clean": sisdr_clean,
            "sisdr_improvement_db": sisdr_clean - sisdr_noisy,
            "noise_reduction_db": nr_db,
            "speech_level_preserved": speech_preserved,
        }

        if not args.no_batch:
            batch_48k = batch_enhance(noisy_48k)
            b_a, c_c = align(clean_48k, batch_48k)
            row["sisdr_batch_vs_clean"] = si_sdr(b_a, c_c)
            sb_a, sb_b = align(batch_48k, out_48k)
            row["sisdr_stream_vs_batch"] = si_sdr(sb_a, sb_b)

        rows.append(row)
        total_dur += row["dur_s"]
        total_rt += row["rtf"] * row["dur_s"]

        msg = (f"  {noisy_p.name:<28} dur={row['dur_s']:5.2f}s "
               f"RTF={rtf:.3f} SI-SDR(noisy={sisdr_noisy:5.2f}, "
               f"stream={sisdr_clean:5.2f}")
        if "sisdr_batch_vs_clean" in row:
            msg += (f", batch={row['sisdr_batch_vs_clean']:5.2f}, "
                    f"stream-vs-batch={row['sisdr_stream_vs_batch']:5.2f}")
        msg += ")"
        print(msg)

    if not rows:
        return 1

    def stats(key: str) -> dict:
        vals = np.array([r[key] for r in rows if key in r])
        return dict(
            mean=float(vals.mean()),
            median=float(np.median(vals)),
            p25=float(np.percentile(vals, 25)),
            p75=float(np.percentile(vals, 75)),
        )

    print(f"\n{'metric':<28}{'mean':>9}{'median':>9}{'p25':>9}{'p75':>9}")
    print("-" * 64)
    keys = [
        "rtf",
        "sisdr_noisy_vs_clean",
        "sisdr_stream_vs_clean",
        "sisdr_improvement_db",
        "noise_reduction_db",
        "speech_level_preserved",
    ]
    if not args.no_batch:
        keys += ["sisdr_batch_vs_clean", "sisdr_stream_vs_batch"]
    summary = {k: stats(k) for k in keys}
    for k, s in summary.items():
        print(f"{k:<28}{s['mean']:>9.3f}{s['median']:>9.3f}{s['p25']:>9.3f}{s['p75']:>9.3f}")

    avg_rtf = total_rt / total_dur if total_dur else float("nan")
    print(f"\nTotal audio: {total_dur:.1f}s   average RTF: {avg_rtf:.3f}   "
          f"~{1/avg_rtf:.0f}× real-time per stream")

    if args.output:
        lines = [
            "# Benchmark results",
            "",
            f"- Files evaluated: **{len(rows)}**",
            f"- Total audio: **{total_dur:.1f} s**",
            f"- Average RTF: **{avg_rtf:.3f}** (~{1/avg_rtf:.0f}× real-time per stream)",
            "",
            "## Summary",
            "",
            "| metric | mean | median | p25 | p75 |",
            "|---|---:|---:|---:|---:|",
        ]
        for k in keys:
            s = summary[k]
            lines.append(f"| `{k}` | {s['mean']:.3f} | {s['median']:.3f} | "
                         f"{s['p25']:.3f} | {s['p75']:.3f} |")
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\nWrote markdown report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
