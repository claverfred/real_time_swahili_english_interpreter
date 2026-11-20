"""Audio loading, validation, resampling, and PCM conversion.

All internal audio is float32 mono in [-1, 1] at ``audio.sample_rate`` (16 kHz by default).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..utils.manifest import FallbackLog


class AudioError(ValueError):
    """Unreadable, malformed, empty, or over-limit audio."""


@dataclass
class AudioInfo:
    path: str
    format: str
    sample_rate_in: int
    channels_in: int
    duration_s: float
    size_mb: float
    decoder: str            # soundfile | ffmpeg

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def ensure_float32_mono(audio: np.ndarray) -> np.ndarray:
    """Validate and normalize an array to contiguous float32 mono."""
    x = np.asarray(audio)
    if x.size == 0:
        raise AudioError("audio is empty")
    if x.ndim == 2:
        # (samples, channels) is the soundfile/sounddevice convention.
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    elif x.ndim != 1:
        raise AudioError(f"expected 1-D or 2-D audio, got shape {x.shape}")
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32, copy=False)
    if not np.all(np.isfinite(x)):
        raise AudioError("audio contains NaN or infinite samples")
    return np.ascontiguousarray(x)


def resample(
    audio: np.ndarray, sr_in: int, sr_out: int, fallbacks: FallbackLog | None = None
) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    if sr_in <= 0 or sr_out <= 0:
        raise AudioError(f"invalid sample rates {sr_in} -> {sr_out}")
    try:
        import soxr

        return soxr.resample(audio, sr_in, sr_out, quality="HQ").astype(np.float32, copy=False)
    except ImportError:
        if fallbacks is not None:
            fallbacks.add("resample", "soxr", "numpy-linear", "soxr not installed; lower quality")
        n_out = int(round(len(audio) * sr_out / sr_in))
        t_out = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(t_out, np.arange(len(audio)), audio).astype(np.float32)


def _decode_ffmpeg(path: Path, sr: int) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError(
            f"{path.suffix or 'this'} format needs FFmpeg, which is not on PATH "
            "(Windows: winget install Gyan.FFmpeg)"
        )
    proc = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, timeout=600,
    )
    if proc.returncode != 0:
        raise AudioError(f"FFmpeg could not decode {path.name}: {proc.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(
    path: str | os.PathLike[str],
    target_sr: int = 16000,
    max_seconds: float | None = None,
    max_mb: float | None = None,
    fallbacks: FallbackLog | None = None,
) -> tuple[np.ndarray, AudioInfo]:
    """Load any supported file as float32 mono at ``target_sr``, enforcing input limits."""
    p = Path(path)
    if not p.is_file():
        raise AudioError(f"audio file not found: {p}")
    size_mb = p.stat().st_size / 1e6
    if max_mb is not None and size_mb > max_mb:
        raise AudioError(f"{p.name} is {size_mb:.1f} MB; limit is {max_mb} MB (audio.max_input_mb)")
    if size_mb == 0:
        raise AudioError(f"{p.name} is empty")

    import soundfile as sf

    try:
        info = sf.info(str(p))
        if max_seconds is not None and info.duration > max_seconds:
            raise AudioError(
                f"{p.name} is {info.duration:.0f} s; limit is {max_seconds} s (audio.max_input_seconds)"
            )
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        audio = ensure_float32_mono(data)
        audio = resample(audio, sr, target_sr, fallbacks)
        meta = AudioInfo(str(p), info.format, sr, info.channels, len(audio) / target_sr, round(size_mb, 3),
                         "soundfile")
    except AudioError:
        raise
    except Exception as sf_error:  # unsupported container (mp3/m4a on old libsndfile, etc.)
        try:
            audio = ensure_float32_mono(_decode_ffmpeg(p, target_sr))
        except AudioError as ff_error:
            raise AudioError(f"cannot read {p.name}: {sf_error}; {ff_error}") from sf_error
        if fallbacks is not None:
            fallbacks.add("decode", "soundfile", "ffmpeg", f"soundfile failed: {sf_error}")
        duration = len(audio) / target_sr
        if max_seconds is not None and duration > max_seconds:
            raise AudioError(f"{p.name} is {duration:.0f} s; limit is {max_seconds} s") from None
        meta = AudioInfo(str(p), p.suffix.lstrip(".").upper(), target_sr, 1, duration, round(size_mb, 3), "ffmpeg")
    return audio, meta


def save_wav(path: str | os.PathLike[str], audio: np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), np.clip(ensure_float32_mono(audio), -1.0, 1.0), sample_rate, subtype="PCM_16")
    return p


def float_to_pcm16(audio: np.ndarray) -> bytes:
    x = np.clip(audio, -1.0, 1.0)
    return (x * 32767.0).round().astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise AudioError("PCM16 byte length must be even")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0
