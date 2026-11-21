"""Post-processing that makes every backend's output consistent and click-free."""

from __future__ import annotations

import numpy as np

from ..audio.io import resample
from ..audio.levels import analyze_levels, dbfs

_EPS = 1e-10


def trim_silence(audio: np.ndarray, sr: int, threshold_dbfs: float = -45.0, keep_ms: float = 30.0) -> np.ndarray:
    """Remove leading/trailing near-silence (backends pad differently), keeping a short margin."""
    if audio.size == 0:
        return audio
    frame = max(1, sr // 100)  # 10 ms
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(np.square(audio[: n * frame].reshape(n, frame), dtype=np.float64), axis=1))
    loud = np.nonzero(20 * np.log10(rms + _EPS) > threshold_dbfs)[0]
    if loud.size == 0:
        return audio[:0]
    keep = int(sr * keep_ms / 1000)
    start = max(0, loud[0] * frame - keep)
    end = min(len(audio), (loud[-1] + 1) * frame + keep)
    return audio[start:end]


def apply_fades(audio: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    n = min(int(sr * fade_ms / 1000), len(audio) // 2)
    if n <= 0:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def normalize_rms(audio: np.ndarray, target_dbfs: float, peak_limit_dbfs: float = -1.0) -> np.ndarray:
    """Scale to a target RMS, never letting the peak exceed ``peak_limit_dbfs``."""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-6:
        return audio
    gain = 10 ** ((target_dbfs - dbfs(rms)) / 20)
    max_gain = 10 ** (peak_limit_dbfs / 20) / max(peak, _EPS)
    return (audio * min(gain, max_gain)).astype(np.float32)


def postprocess(audio: np.ndarray, sr_in: int, sr_out: int, *, trim: bool, fade_ms: float,
                target_rms_dbfs: float | None) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr_in != sr_out:
        audio = resample(audio, sr_in, sr_out)
    if trim:
        audio = trim_silence(audio, sr_out)
    if target_rms_dbfs is not None:
        audio = normalize_rms(audio, target_rms_dbfs)
    return apply_fades(audio, sr_out, fade_ms)


def validate_output(audio: np.ndarray, sr: int, text: str) -> list[str]:
    """Problems a listener would notice. Empty list = OK."""
    if audio.size == 0:
        return ["empty_audio"]
    lv = analyze_levels(audio, sr)
    warnings = []
    if lv["clipping_ratio"] > 0.001:
        warnings.append("clipping")
    if lv["rms_dbfs"] < -40:
        warnings.append("too_quiet")
    if lv["silence_ratio"] > 0.6:
        warnings.append("mostly_silent")
    # Rough speaking-rate sanity check: English TTS is ~8-25 characters per second.
    cps = len(text) / max(lv["duration_s"], 1e-3)
    if len(text) >= 20 and not 4 <= cps <= 40:
        warnings.append(f"implausible_duration ({cps:.0f} chars/s)")
    return warnings
