"""Per-frame streaming DeepFilterNet3 for Pipecat.

The :class:`DeepFilterNetStreamFilter` is a drop-in ``BaseAudioFilter`` that
runs DFN3 with one-hop-in, one-hop-out streaming inference (~50 ms total
filter delay) using Sonos' ``tract`` pulsed ONNX runtime.

For lower-level access, :class:`PerFrameDfn` exposes the bare engine that
ingests 480-sample 48 kHz hops and emits 480-sample 48 kHz hops.
"""

from .engine import (
    DFN_SAMPLE_RATE,
    HOP_SIZE,
    PerFrameDfn,
    ensure_runnables,
)
from .filter import DeepFilterNetStreamFilter

__all__ = [
    "DFN_SAMPLE_RATE",
    "HOP_SIZE",
    "PerFrameDfn",
    "DeepFilterNetStreamFilter",
    "ensure_runnables",
]

__version__ = "0.1.0"
