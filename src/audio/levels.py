"""Level and format validation: loudness, clipping, silence, DC offset."""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def dbfs(x: float) -> float:
    return float(20.0 * np.log10(max(x, _EPS)))


def analyze_levels(audio: np.ndarray, sample_rate: int, frame_ms: int = 30,
                   silence_dbfs: float = -50.0) -> dict[str, object]:
    n = len(audio)
    duration = n / sample_rate if sample_rate else 0.0
    if n == 0:
        return {"duration_s": 0.0, "warnings": ["empty audio"]}
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    clip_ratio = float(np.mean(np.abs(audio) >= 0.999))
    frame = max(1, sample_rate * frame_ms // 1000)
    n_frames = max(1, n // frame)
    frames = audio[: n_frames * frame].reshape(n_frames, -1) if n >= frame else audio.reshape(1, -1)
    frame_db = 20 * np.log10(np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1)) + _EPS)
    silence_ratio = float(np.mean(frame_db < silence_dbfs))

    warnings = []
    if duration < 0.3:
        warnings.append(f"very short input ({duration:.2f} s)")
    if silence_ratio > 0.98:
        warnings.append("input is (almost) entirely silent")
    elif dbfs(rms) < -45:
        warnings.append(f"very quiet input (RMS {dbfs(rms):.1f} dBFS); check microphone gain")
    if clip_ratio > 0.001:
        warnings.append(f"clipping in {clip_ratio:.2%} of samples; reduce input gain")
    dc = float(np.mean(audio))
    if abs(dc) > 0.05:
        warnings.append(f"DC offset {dc:+.3f}")
    return {
        "duration_s": round(duration, 3),
        "sample_rate": sample_rate,
        "rms_dbfs": round(dbfs(rms), 2),
        "peak_dbfs": round(dbfs(peak), 2),
        "clipping_ratio": round(clip_ratio, 6),
        "silence_ratio": round(silence_ratio, 4),
        "dc_offset": round(dc, 5),
        "warnings": warnings,
    }
