"""Simulated live input: stream an audio array/file in fixed blocks, optionally at real-time pace."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np


async def stream_array(audio: np.ndarray, sample_rate: int = 16000, block_ms: int = 100,
                       realtime_factor: float = 1.0) -> AsyncIterator[np.ndarray]:
    """Yield ``block_ms`` chunks. ``realtime_factor`` 1.0 = wall-clock pace, 0 = as fast as possible.

    Pacing is anchored to the start time, so slow consumers do not accumulate drift.
    """
    step = max(1, int(sample_rate * block_ms / 1000))
    t0 = time.perf_counter()
    for i, start in enumerate(range(0, len(audio), step)):
        if realtime_factor > 0:
            due = t0 + (i * step / sample_rate) / realtime_factor
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)
        yield audio[start:start + step]
