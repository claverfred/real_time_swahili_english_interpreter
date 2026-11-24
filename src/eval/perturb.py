"""Deterministic audio perturbations for robustness testing (spec 14: noisy, quiet, fast/slow,
reverberant speech, poor microphones).

These are *simulated* conditions applied to clean recordings. They show relative sensitivity,
not performance in a real noisy room. Every condition is seeded per item, so runs repeat exactly.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np

SR = 16000


def _rng(item_id: str, condition: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{condition}:{item_id}".encode()))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def _limit(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return (x / peak * 0.99 if peak > 0.99 else x).astype(np.float32)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise by spectral shaping; closer to room/crowd noise than white noise."""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / (_rms(x) or 1.0)


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Pink noise at ``snr_db`` relative to the speech RMS (speech-active frames only)."""
    frame = SR // 50
    frames = [x[i:i + frame] for i in range(0, len(x) - frame + 1, frame)] or [x]
    levels = sorted((_rms(f) for f in frames), reverse=True)
    speech_rms = float(np.mean(levels[: max(1, len(levels) // 2)]))   # loudest half ~ speech
    noise = _pink(len(x), rng) * speech_rms / (10 ** (snr_db / 20))
    return _limit(x + noise)


def gain(x: np.ndarray, db: float) -> np.ndarray:
    return _limit(x * (10 ** (db / 20)))


def reverb(x: np.ndarray, rt60_s: float, rng: np.random.Generator) -> np.ndarray:
    """Synthetic room: exponentially decaying noise impulse response (RT60), direct path kept."""
    n = int(rt60_s * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60_s)      # -60 dB at rt60
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    y = np.convolve(x, ir)[: len(x)]
    return _limit(y * (_rms(x) / (_rms(y) or 1.0)))


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    """Tempo and pitch change by resampling (like a faster or slower speaker, pitch shifts too)."""
    from ..audio.io import resample

    return resample(x.astype(np.float32), int(round(SR * factor)), SR).astype(np.float32)


def telephone(x: np.ndarray) -> np.ndarray:
    """Narrowband 300-3400 Hz at 8 kHz with 8-bit mu-law, like a poor phone or cheap headset."""
    from ..audio.io import resample

    y = resample(x.astype(np.float32), SR, 8000)
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / 8000)
    spec[(f < 300) | (f > 3400)] = 0
    y = np.fft.irfft(spec, len(y))
    mu = 255.0
    q = np.sign(y) * np.log1p(mu * np.clip(np.abs(y), 0, 1)) / np.log1p(mu)
    q = np.round(q * 127) / 127
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return _limit(resample(y.astype(np.float32), 8000, SR))


CONDITIONS: dict[str, tuple[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]] = {
    "clean": ("original recording", lambda x, r: x),
    "noise_20db": ("pink noise, SNR 20 dB (quiet office)", lambda x, r: add_noise(x, 20, r)),
    "noise_10db": ("pink noise, SNR 10 dB (busy room)", lambda x, r: add_noise(x, 10, r)),
    "noise_5db": ("pink noise, SNR 5 dB (loud)", lambda x, r: add_noise(x, 5, r)),
    "quiet_-20db": ("20 dB quieter (distant or soft speaker)", lambda x, r: gain(x, -20)),
    "reverb_0.6s": ("synthetic room reverb, RT60 0.6 s", lambda x, r: reverb(x, 0.6, r)),
    "fast_1.15x": ("15% faster (pitch up)", lambda x, r: speed(x, 1.15)),
    "slow_0.85x": ("15% slower (pitch down)", lambda x, r: speed(x, 0.85)),
    "telephone": ("300-3400 Hz, 8 kHz mu-law (poor microphone)", lambda x, r: telephone(x)),
}


def apply(condition: str, audio: np.ndarray, item_id: str) -> np.ndarray:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {', '.join(CONDITIONS)}")
    return CONDITIONS[condition][1](np.asarray(audio, np.float32), _rng(item_id, condition))
