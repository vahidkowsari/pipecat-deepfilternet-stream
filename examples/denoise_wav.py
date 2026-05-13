"""Denoise a wav file end-to-end with the per-frame engine.

Usage:
    python denoise_wav.py noisy.wav out.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from pipecat_deepfilternet_stream import HOP_SIZE, PerFrameDfn
from pipecat_deepfilternet_stream.engine import DFN_SAMPLE_RATE


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()

    audio, sr = sf.read(str(args.input), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != DFN_SAMPLE_RATE:
        a48 = soxr.resample(audio, sr, DFN_SAMPLE_RATE, quality="HQ").astype(np.float32)
    else:
        a48 = audio
    n = (a48.size // HOP_SIZE) * HOP_SIZE
    a48 = a48[:n]

    engine = PerFrameDfn()
    t0 = time.perf_counter()
    out_chunks = [engine.process_hop(a48[i : i + HOP_SIZE]) for i in range(0, n, HOP_SIZE)]
    elapsed = time.perf_counter() - t0
    out48 = np.concatenate(out_chunks)
    out = soxr.resample(out48, DFN_SAMPLE_RATE, sr, quality="HQ").astype(np.float32) if sr != DFN_SAMPLE_RATE else out48
    sf.write(str(args.output), out, sr, subtype="PCM_16")

    rtf = elapsed / (n / DFN_SAMPLE_RATE)
    print(
        f"{args.input.name}: sr={sr} dur={n / DFN_SAMPLE_RATE:.2f}s "
        f"hops={n // HOP_SIZE} elapsed={elapsed * 1000:.0f}ms RTF={rtf:.3f}  "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
