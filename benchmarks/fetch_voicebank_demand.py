"""Fetch a subset of VoiceBank-DEMAND-16k from HuggingFace as paired wavs.

VoiceBank-DEMAND is the standard DFN3 benchmark dataset (noisy + clean pairs).
The 16 kHz variant lives at huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k.

Writes a directory layout compatible with benchmark.py:

    <out>/
    ├── noisy/p232_005.wav
    └── clean/p232_005.wav

Usage:
    pip install datasets soundfile
    python fetch_voicebank_demand.py ./vbd16k --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--split", default="test")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Need: pip install datasets", file=sys.stderr)
        return 1

    noisy_dir = args.out / "noisy"
    clean_dir = args.out / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split=args.split, streaming=True)
    saved = 0
    for ex in ds:
        if saved >= args.limit:
            break
        name = ex.get("filename") or f"sample_{saved:04d}"
        if not name.endswith(".wav"):
            name = f"{Path(name).stem}.wav"
        noisy = ex["noisy"]["array"]
        clean = ex["clean"]["array"]
        sr = ex["noisy"]["sampling_rate"]
        sf.write(str(noisy_dir / name), np.asarray(noisy, dtype=np.float32), sr, subtype="PCM_16")
        sf.write(str(clean_dir / name), np.asarray(clean, dtype=np.float32), sr, subtype="PCM_16")
        saved += 1

    print(f"Saved {saved} paired files to {args.out}/{{noisy,clean}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
