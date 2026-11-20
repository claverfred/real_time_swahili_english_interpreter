"""Live microphone capture as an async stream of float32 mono chunks.

Opens the device at the pipeline rate when the driver allows it; otherwise opens at the device's
native rate and resamples in a streaming (stateful) way. Either path is recorded. Overflows from
the driver and drops from a full queue are counted, never hidden.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.manifest import FallbackLog

CONSENT_NOTICE = (
    "Microphone capture is starting. Audio is processed locally and is not saved unless you "
    "asked for it (e.g. --output). Make sure everyone being recorded has agreed to it."
)


class CaptureError(RuntimeError):
    pass


def _sd() -> Any:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise CaptureError(f"sounddevice/PortAudio unavailable: {exc} (pip install -e \".[local]\")") from exc
    return sd


def list_devices() -> list[dict[str, Any]]:
    sd = _sd()
    hostapis = sd.query_hostapis()
    default_in, default_out = sd.default.device
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({
            "index": i,
            "name": d["name"],
            "hostapi": hostapis[d["hostapi"]]["name"],
            "max_input_channels": d["max_input_channels"],
            "max_output_channels": d["max_output_channels"],
            "default_samplerate": d["default_samplerate"],
            "default_input": i == default_in,
            "default_output": i == default_out,
        })
    return out


@dataclass
class CaptureStats:
    blocks: int = 0
    samples: int = 0
    driver_overflows: int = 0
    queue_drops: int = 0
    device_rate: int = 0
    resampled: bool = False
    notes: list[str] = field(default_factory=list)


class MicrophoneStream:
    """``async with MicrophoneStream(...) as mic: async for chunk in mic: ...``"""

    def __init__(self, device: int | str | None = None, sample_rate: int = 16000, block_ms: int = 100,
                 queue_max: int = 50, fallbacks: FallbackLog | None = None) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.queue_max = queue_max
        self.fallbacks = fallbacks
        self.stats = CaptureStats()
        self._stream: Any = None
        self._queue: asyncio.Queue[np.ndarray | None] | None = None
        self._resampler: Any = None

    def _open(self, rate: int) -> Any:
        sd = _sd()
        blocksize = int(rate * self.block_ms / 1000)
        return sd.InputStream(device=self.device, channels=1, samplerate=rate, dtype="float32",
                              blocksize=blocksize, callback=self._callback)

    async def start(self) -> MicrophoneStream:
        sd = _sd()
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(self.queue_max)
        try:
            self._stream = self._open(self.sample_rate)
            self.stats.device_rate = self.sample_rate
        except sd.PortAudioError as exc:
            info = sd.query_devices(self.device, "input")
            native = int(info["default_samplerate"])
            import soxr

            self._resampler = soxr.ResampleStream(native, self.sample_rate, 1, dtype="float32")
            self._stream = self._open(native)
            self.stats.device_rate, self.stats.resampled = native, True
            if self.fallbacks is not None:
                self.fallbacks.add("capture", f"{self.sample_rate} Hz", f"{native} Hz + resample", str(exc))
        self._stream.start()
        return self

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status and status.input_overflow:
            self.stats.driver_overflows += 1
        chunk = indata[:, 0].copy()
        if self._resampler is not None:
            chunk = self._resampler.resample_chunk(chunk)
        self._loop.call_soon_threadsafe(self._put, chunk)

    def _put(self, chunk: np.ndarray) -> None:
        assert self._queue is not None
        if self._queue.full():
            self._queue.get_nowait()
            self.stats.queue_drops += 1
        self._queue.put_nowait(chunk)
        self.stats.blocks += 1
        self.stats.samples += len(chunk)

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._queue is not None:
            if self._queue.full():
                self._queue.get_nowait()
            self._queue.put_nowait(None)

    async def __aenter__(self) -> MicrophoneStream:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def __aiter__(self) -> AsyncIterator[np.ndarray]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[np.ndarray]:
        if self._queue is None:
            raise CaptureError("call start() (or use 'async with') before iterating")
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            if len(chunk):
                yield chunk
