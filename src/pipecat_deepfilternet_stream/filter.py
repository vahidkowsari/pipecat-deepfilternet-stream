"""Pipecat ``BaseAudioFilter`` wrapping the per-frame DFN3 engine.

Wire it onto an input transport:

    from pipecat_deepfilternet_stream import DeepFilterNetStreamFilter

    transport = FastAPIWebsocketTransport(
        websocket=ws,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_filter=DeepFilterNetStreamFilter(),
            ...
        ),
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

from .engine import DFN_SAMPLE_RATE, HOP_SIZE, PerFrameDfn, ensure_runnables


class DeepFilterNetStreamFilter(BaseAudioFilter):
    """Per-frame streaming DeepFilterNet3 noise suppression for Pipecat.

    Each call is a single 10 ms hop @ 48 kHz processed through tract's
    pulsed ONNX runtime. Total filter delay ≈ 50 ms (matches Rikorose's
    Rust binary), vs ~170 ms for chunked-batch DFN3 streaming.

    Args:
        resampler_quality: SOXR quality if the transport rate isn't 48 kHz.
            "QQ" is the lowest-latency option (matches pipecat's RNNoise).
    """

    def __init__(self, resampler_quality: str = "QQ") -> None:
        self._enabled = True
        self._ready = False
        self._sample_rate = 0
        self._resampler_quality = resampler_quality

        self._engine: PerFrameDfn | None = None
        self._buf48k = np.zeros(0, dtype=np.float32)
        self._resampler_in: Any = None
        self._resampler_out: Any = None

    async def start(self, sample_rate: int) -> None:
        try:
            ensure_runnables()
        except Exception as e:
            logger.error(
                f"Could not load pulsed DFN3 ONNX runnables ({e}). "
                "Install dependencies: pip install pipecat-deepfilternet-stream[all]"
            )
            self._ready = False
            return

        try:
            self._engine = PerFrameDfn()
        except Exception as e:
            logger.error(f"Could not initialise per-frame DFN3 engine: {e}")
            self._ready = False
            return

        self._sample_rate = sample_rate
        if sample_rate != DFN_SAMPLE_RATE:
            try:
                from pipecat.audio.resamplers.soxr_stream_resampler import (
                    SOXRStreamAudioResampler,
                )

                self._resampler_in = SOXRStreamAudioResampler(quality=self._resampler_quality)
                self._resampler_out = SOXRStreamAudioResampler(quality=self._resampler_quality)
            except ImportError as e:
                logger.error(f"Could not import SOXRStreamAudioResampler: {e}")
                self._ready = False
                return

        self._ready = True
        logger.info(
            f"DeepFilterNetStreamFilter ready (transport={sample_rate} Hz, "
            f"model={DFN_SAMPLE_RATE} Hz, hop={HOP_SIZE} samples)"
        )

    async def stop(self) -> None:
        self._engine = None
        self._resampler_in = None
        self._resampler_out = None
        self._buf48k = np.zeros(0, dtype=np.float32)
        self._ready = False

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._ready or not self._enabled or not audio or self._engine is None:
            return audio

        # 1. Resample to 48 kHz if needed.
        if self._sample_rate != DFN_SAMPLE_RATE and self._resampler_in is not None:
            in_bytes = await self._resampler_in.resample(
                audio, self._sample_rate, DFN_SAMPLE_RATE
            )
        else:
            in_bytes = audio
        if not in_bytes:
            return b""

        in_f = np.frombuffer(in_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf48k = np.concatenate([self._buf48k, in_f])

        # 2. Drain whole hops.
        out_pieces: list[np.ndarray] = []
        while self._buf48k.size >= HOP_SIZE:
            hop = self._buf48k[:HOP_SIZE]
            self._buf48k = self._buf48k[HOP_SIZE:]
            out_pieces.append(self._engine.process_hop(hop))

        if not out_pieces:
            return b""

        out_48k_f = np.concatenate(out_pieces)
        out_48k_i16 = np.clip(out_48k_f * 32767.0, -32768, 32767).astype(np.int16).tobytes()

        # 3. Resample back to transport rate.
        if self._sample_rate != DFN_SAMPLE_RATE and self._resampler_out is not None:
            return await self._resampler_out.resample(
                out_48k_i16, DFN_SAMPLE_RATE, self._sample_rate
            )
        return out_48k_i16
