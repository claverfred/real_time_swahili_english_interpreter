"""Fixed-size framing of arbitrarily sized audio chunks."""

from __future__ import annotations

import numpy as np


class FrameChunker:
    """Accepts chunks of any length and returns complete frames of ``frame_samples``."""

    def __init__(self, frame_samples: int) -> None:
        if frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        self.frame_samples = frame_samples
        self._residual = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        data = np.concatenate([self._residual, chunk.astype(np.float32, copy=False)])
        n = len(data) // self.frame_samples
        frames = [data[i * self.frame_samples:(i + 1) * self.frame_samples] for i in range(n)]
        self._residual = data[n * self.frame_samples:].copy()
        return frames

    def flush(self) -> np.ndarray | None:
        """Return the zero-padded final partial frame, if any."""
        if len(self._residual) == 0:
            return None
        frame = np.zeros(self.frame_samples, dtype=np.float32)
        frame[: len(self._residual)] = self._residual
        self._residual = np.zeros(0, dtype=np.float32)
        return frame

    @property
    def pending_samples(self) -> int:
        return len(self._residual)

    def reset(self) -> None:
        self._residual = np.zeros(0, dtype=np.float32)
