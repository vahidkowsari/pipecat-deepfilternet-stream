"""Smoke tests for the per-frame DFN3 engine.

The deep validation against ``df.enhance.enhance()`` (45 dB SI-SDR identity)
lives in the development repo; these tests just verify the engine runs
without errors, produces the right output shape, and applies non-trivial
attenuation to a noisy signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipecat_deepfilternet_stream import PerFrameDfn
from pipecat_deepfilternet_stream.engine import HOP_SIZE


def test_engine_shape_contract() -> None:
    """One hop in, one hop out, same shape and float32 dtype."""
    eng = PerFrameDfn()
    hop = np.zeros(HOP_SIZE, dtype=np.float32)
    out = eng.process_hop(hop)
    assert out.shape == (HOP_SIZE,)
    assert out.dtype == np.float32


def test_engine_rejects_wrong_size() -> None:
    eng = PerFrameDfn()
    with pytest.raises(ValueError):
        eng.process_hop(np.zeros(HOP_SIZE - 1, dtype=np.float32))


def test_engine_attenuates_white_noise() -> None:
    """5 seconds of white noise should come out quieter than it went in."""
    rng = np.random.default_rng(seed=0)
    sr = 48_000
    audio = rng.standard_normal(sr * 5).astype(np.float32) * 0.1
    audio = audio[: (audio.size // HOP_SIZE) * HOP_SIZE]

    eng = PerFrameDfn()
    out_pieces = [
        eng.process_hop(audio[i : i + HOP_SIZE]) for i in range(0, audio.size, HOP_SIZE)
    ]
    out = np.concatenate(out_pieces)
    # Skip the first 100 ms (model warm-up region — zero-initialised ring
    # buffers and feature state).
    warmup = sr // 10
    in_rms = float(np.sqrt((audio[warmup:] ** 2).mean()))
    out_rms = float(np.sqrt((out[warmup:] ** 2).mean()))
    # White noise is pure noise; the model should suppress it heavily.
    assert out_rms < in_rms * 0.5, f"expected ≥6 dB attenuation, got in_rms={in_rms:.4f} out_rms={out_rms:.4f}"


def test_two_engines_have_independent_state() -> None:
    """Constructing two PerFrameDfn instances must not share mutable state."""
    eng_a = PerFrameDfn()
    eng_b = PerFrameDfn()
    audio = np.random.default_rng(seed=42).standard_normal(HOP_SIZE).astype(np.float32) * 0.1

    out_a = eng_a.process_hop(audio.copy())
    # eng_b should produce the same first-hop output as eng_a (both fresh state).
    out_b = eng_b.process_hop(audio.copy())
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)

    # Now feed eng_a a second hop. eng_b should be unchanged.
    eng_a.process_hop(audio.copy())
    out_b_again = eng_b.process_hop(audio.copy())
    # eng_b's second-hop output should match eng_a's second hop, NOT third.
    # We just verify eng_b is still consistent with a fresh second-hop call.
    eng_c = PerFrameDfn()
    eng_c.process_hop(audio.copy())
    out_c = eng_c.process_hop(audio.copy())
    np.testing.assert_allclose(out_b_again, out_c, atol=1e-6)
