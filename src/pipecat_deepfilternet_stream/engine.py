"""Per-frame streaming DeepFilterNet3 inference engine.

A Python port of the per-frame inference loop in Rikorose's
``libDF/src/tract.rs`` (the Rust binary that ships with DeepFilterNet).
Processes one hop (10 ms @ 48 kHz) of audio per call with batch-equivalent
denoising quality. Total filter delay ≈ 50 ms (40 ms model lookahead +
10 ms STFT pre-roll), versus ~170 ms for chunked offline-style streaming.

The two non-obvious ingredients:

1. **`tract` Python bindings with `pulse(S, 1)`** — Sonos' tract auto-rewrites
   the ONNX models for streaming: internal state for both GRUs AND conv2d
   time-kernels is managed transparently across `state.run()` calls. This is
   exactly the same transformation Rikorose's Rust binary applies, and it's
   what onnxruntime cannot do.

2. **`libdf.DF.analysis(input, reset=False)`** — the libdf Python binding
   defaults ``reset=None`` which silently resets the streaming STFT state
   between per-hop calls. ``reset=False`` keeps OLA continuous and makes
   per-hop output bit-exact identical to a batch call on the same audio.

The class exposes a single ``process_hop(samples)`` method. Resampling
between an arbitrary transport sample rate and DFN3's native 48 kHz is up
to the caller; see ``pipecat_deepfilternet_stream.filter`` for a Pipecat
``BaseAudioFilter`` that wires SOXR resampling onto this engine.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


# DFN3 model constants. Verified against the bundled ONNX exports.
DFN_SAMPLE_RATE = 48_000
FFT_SIZE = 960
HOP_SIZE = 480
N_FREQS = FFT_SIZE // 2 + 1  # 481
NB_DF = 96
NB_ERB = 32
DF_ORDER = 5
DF_LOOKAHEAD = 2
CONV_LOOKAHEAD = 2
ALPHA = 0.99  # df.utils.get_norm_alpha(False)

# Initial feature-normalization state vectors. From libDF/src/lib.rs:
#   pub const MEAN_NORM_INIT: [f32; 2] = [-60., -90.];
#   pub const UNIT_NORM_INIT: [f32; 2] = [0.001, 0.0001];
MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)


ONNX_MODELS_DIR = Path(__file__).parent / "onnx_models"


_runnables: dict[str, Any] = {}


def ensure_runnables(onnx_dir: Path | None = None) -> None:
    """Load and pulse the three ONNX submodels once, process-wide.

    Idempotent. Pulsing/optimisation takes ~1 s on Apple Silicon and is shared
    across every ``PerFrameDfn`` instance. Per-stream state is created in
    each instance via ``runnable.spawn_state()``.
    """
    if _runnables:
        return
    import tract

    src = onnx_dir or ONNX_MODELS_DIR
    o = tract.onnx()
    for name in ("enc", "erb_dec", "df_dec"):
        m = (
            o.model_for_path(str(src / f"{name}.onnx"))
            .into_typed()
            .into_decluttered()
        )
        m.pulse("S", 1)
        _runnables[name] = m.into_optimized().into_runnable()


def _build_erb_inv_fb(erb_widths: np.ndarray) -> np.ndarray:
    """Construct the ERB → linear inverse filterbank.

    Mirrors ``df.modules.erb_fb(..., inverse=True)``: each ERB band is mapped
    back to its linear-frequency bins with uniform 1/width weight. Returns a
    [nb_erb, n_freqs] float32 matrix such that ``mask @ erb_inv_fb`` produces
    per-bin gains.
    """
    nb_erb = len(erb_widths)
    fb = np.zeros((nb_erb, N_FREQS), dtype=np.float32)
    bin_offset = 0
    for b, w in enumerate(erb_widths):
        w = int(w)
        # Each erb band b spans `w` linear bins starting at `bin_offset`.
        fb[b, bin_offset : bin_offset + w] = 1.0
        bin_offset += w
    # Normalise so each linear bin sums to 1 across erb bands (one-hot here,
    # so this is effectively a no-op for the inverse direction). Kept for
    # clarity / future divergence from one-hot membership.
    col_sums = fb.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    fb = fb / col_sums
    return fb


class PerFrameDfn:
    """Per-hop streaming DeepFilterNet3 inference.

    Construct one instance per audio stream. Per-call state (libdf STFT
    buffers, feature normalization state, tract pulsed-model state, spec
    ring buffers) is per-instance; the heavy pulsed ONNX runnables are
    process-wide singletons.

    Args:
        onnx_dir: Override the bundled ONNX directory. Defaults to the
            ``onnx_models/`` folder shipped with this package.
    """

    def __init__(self, onnx_dir: Path | None = None) -> None:
        import libdf
        self._libdf = libdf

        ensure_runnables(onnx_dir)
        self._enc_state = _runnables["enc"].spawn_state()
        self._erb_dec_state = _runnables["erb_dec"].spawn_state()
        self._df_dec_state = _runnables["df_dec"].spawn_state()

        self._df_state = libdf.DF(
            DFN_SAMPLE_RATE, FFT_SIZE, HOP_SIZE, nb_bands=NB_ERB, min_nb_erb_freqs=2
        )
        self._erb_widths = self._df_state.erb_widths()
        self._erb_inv_fb = _build_erb_inv_fb(self._erb_widths)

        # Feature-norm state — managed in Python because libdf's Python
        # bindings don't mutate the state arg. Formulas from libDF/src/lib.rs:
        #   band_mean_norm_erb: s = x*(1-α) + s*α;  x_norm = (x - s) / 40
        #   band_unit_norm:     s = |x|*(1-α) + s*α; x_norm = x / sqrt(s)
        self._erb_norm_state = np.linspace(
            MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], NB_ERB, dtype=np.float32
        )
        self._unit_norm_state = np.linspace(
            UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], NB_DF, dtype=np.float32
        )

        # Ring buffers. ``_rolling_y`` holds the spec frames whose mask target
        # is at index ``df_order - 1`` (= 2 behind newest, matching the
        # encoder's ``conv_lookahead = 2``). ``_rolling_x`` holds noisy spec
        # for the df_op multi-frame filter.
        self._rolling_y: deque[np.ndarray] = deque(
            [np.zeros(N_FREQS, dtype=np.complex64) for _ in range(DF_ORDER + CONV_LOOKAHEAD)],
            maxlen=DF_ORDER + CONV_LOOKAHEAD,
        )
        self._rolling_x: deque[np.ndarray] = deque(
            [np.zeros(N_FREQS, dtype=np.complex64) for _ in range(DF_ORDER)],
            maxlen=DF_ORDER,
        )
        self._target_idx = DF_ORDER - 1

    def process_hop(self, hop_audio: np.ndarray) -> np.ndarray:
        """Process one hop (``HOP_SIZE`` samples at 48 kHz). Returns ``HOP_SIZE``
        samples of enhanced audio with ~30 ms algorithmic delay relative to
        the input.
        """
        if hop_audio.shape != (HOP_SIZE,):
            raise ValueError(f"expected {HOP_SIZE}-sample hop, got shape {hop_audio.shape}")

        # 1. STFT analysis. reset=False keeps OLA state continuous across calls.
        spec = self._df_state.analysis(
            hop_audio.astype(np.float32, copy=False)[None], reset=False
        )
        spec_f = spec[0, 0]

        # 2. Push to ring buffers.
        self._rolling_y.append(spec_f.copy())
        self._rolling_x.append(spec_f.copy())

        # 3. Streaming feature normalization (formulas from libDF/src/lib.rs).
        erb_mag = self._libdf.erb(spec, self._erb_widths)  # [1, 1, NB_ERB] dB
        erb_vec = erb_mag[0, 0]
        self._erb_norm_state = erb_vec * (1.0 - ALPHA) + self._erb_norm_state * ALPHA
        feat_erb = (erb_vec - self._erb_norm_state) / 40.0
        feat_erb_in = feat_erb.reshape(1, 1, 1, NB_ERB).astype(np.float32, copy=False)

        spec_df_vec = spec[0, 0, :NB_DF]
        mag = np.abs(spec_df_vec).astype(np.float32)
        self._unit_norm_state = mag * (1.0 - ALPHA) + self._unit_norm_state * ALPHA
        feat_spec_c = spec_df_vec / np.sqrt(self._unit_norm_state)
        feat_spec_in = np.empty((1, 2, 1, NB_DF), dtype=np.float32)
        feat_spec_in[0, 0, 0] = feat_spec_c.real
        feat_spec_in[0, 1, 0] = feat_spec_c.imag

        # 4. Encoder (pulsed → internal GRU + conv state persists across calls).
        enc_out = self._enc_state.run([feat_erb_in, feat_spec_in])
        e0 = enc_out[0].to_numpy()
        e1 = enc_out[1].to_numpy()
        e2 = enc_out[2].to_numpy()
        e3 = enc_out[3].to_numpy()
        emb = enc_out[4].to_numpy()
        c0 = enc_out[5].to_numpy()

        # 5. ERB decoder (pulsed) → mask.
        m = self._erb_dec_state.run([emb, e3, e2, e1, e0])[0].to_numpy()

        # 6. DF decoder (pulsed) → multi-frame filter coefficients.
        coefs = self._df_dec_state.run([emb, c0])[0].to_numpy()

        # 7. Mask target frame (df_order - 1 in _rolling_y).
        target = self._rolling_y[self._target_idx].copy()

        # 8. ERB mask → per-bin gain.
        gain = m[0, 0, 0] @ self._erb_inv_fb
        spec_m = target * gain

        # 9. df_op on noisy _rolling_x: filter the lowest NB_DF bins by a
        # complex multiply-and-sum across the DF_ORDER recent frames.
        cflat = coefs[0, 0].reshape(NB_DF, DF_ORDER, 2)
        coefs_c = cflat[..., 0].astype(np.complex64) + 1j * cflat[..., 1].astype(np.float32)
        window = np.empty((DF_ORDER, NB_DF), dtype=np.complex64)
        for k in range(DF_ORDER):
            window[k] = self._rolling_x[k][:NB_DF]
        spec_filtered = (window * coefs_c.T).sum(axis=0)

        # 10. Combine: lowest NB_DF bins from df_op, upper bins from mask.
        enhanced = spec_m
        enhanced[:NB_DF] = spec_filtered

        # 11. ISTFT (reset=False keeps OLA continuous).
        return self._df_state.synthesis(
            enhanced[None, None].astype(np.complex64), reset=False
        )[0]
