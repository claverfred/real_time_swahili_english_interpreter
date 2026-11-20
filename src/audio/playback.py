"""Local speaker playback (used by the live CLI/UI; browsers and APIs stream audio instead)."""

from __future__ import annotations

import numpy as np

from ..capture.microphone import _sd


def play(audio: np.ndarray, sample_rate: int, device: int | str | None = None, blocking: bool = True) -> None:
    sd = _sd()
    sd.play(np.clip(audio, -1.0, 1.0).astype(np.float32), samplerate=sample_rate, device=device,
            blocking=blocking)


def stop() -> None:
    _sd().stop()
