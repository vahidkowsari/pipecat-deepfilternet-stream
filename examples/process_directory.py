"""Run the per-frame DFN3 engine on a directory of wavs, write before/after pairs.

Uses the ``pipecat-deepfilternet-stream`` package (now living in its own repo at
https://github.com/vahidkowsari/pipecat-deepfilternet-stream).

Usage:
    python process_perframe.py samples/noisy_testset_wav --out demo_perframe/vctk
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from pipecat_deepfilternet_stream import HOP_SIZE, PerFrameDfn
from pipecat_deepfilternet_stream.engine import DFN_SAMPLE_RATE


def process_file(in_path: Path) -> tuple[np.ndarray, int, float, float]:
    audio, sr = sf.read(str(in_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != DFN_SAMPLE_RATE:
        a48 = soxr.resample(audio, sr, DFN_SAMPLE_RATE, quality="HQ").astype(np.float32)
    else:
        a48 = audio
    n = (a48.size // HOP_SIZE) * HOP_SIZE
    a48 = a48[:n]

    # Fresh engine per file — state shouldn't leak across files.
    engine = PerFrameDfn()
    t0 = time.perf_counter()
    out_chunks = [engine.process_hop(a48[i : i + HOP_SIZE]) for i in range(0, n, HOP_SIZE)]
    elapsed = time.perf_counter() - t0
    out48 = np.concatenate(out_chunks)

    if sr != DFN_SAMPLE_RATE:
        out_native = soxr.resample(out48, DFN_SAMPLE_RATE, sr, quality="HQ").astype(np.float32)
    else:
        out_native = out48
    return out_native, sr, n / DFN_SAMPLE_RATE, elapsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("demo_perframe"))
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--pattern", default="*.wav")
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    excludes = ("_denoised", "_webrtc", "_perframe", "_before", "_after")
    files = sorted(
        f for f in args.input_dir.glob(args.pattern)
        if not any(s in f.stem for s in excludes)
    )[: args.limit]
    if not files:
        print(f"No files matching {args.pattern} under {args.input_dir}")
        return 1

    print(f"Processing {len(files)} files into {out_dir}/")
    print(f"{'file':<32}{'dur':>8}{'rtf':>8}")
    print("-" * 48)
    for src in files:
        out_audio, sr, dur, elapsed = process_file(src)
        rtf = elapsed / dur if dur else float("inf")
        before = out_dir / f"{src.stem}_before.wav"
        after = out_dir / f"{src.stem}_after.wav"
        shutil.copyfile(src, before)
        sf.write(str(after), out_audio, sr, subtype="PCM_16")
        print(f"{src.name:<32}{dur:>7.2f}s{rtf:>8.3f}")

    print(f"\nWrote {len(files) * 2} files to {out_dir.resolve()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
