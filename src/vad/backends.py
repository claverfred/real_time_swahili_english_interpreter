"""Frame-level VAD backends. Each returns a speech probability in [0, 1] per frame.

- ``silero``: Silero VAD v5 via onnxruntime (no PyTorch needed). MIT license.
- ``webrtc``: Google WebRTC VAD (binary decisions). BSD license.
- ``energy``: adaptive-noise-floor energy detector; dependency-free last resort.
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

from ..audio.io import float_to_pcm16
from ..config.schema import VadConfig
from ..utils.manifest import FallbackLog

SILERO_VERSION = "v5.1.2"
SILERO_URL = f"https://github.com/snakers4/silero-vad/raw/{SILERO_VERSION}/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256 = "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"
CHAIN = ["silero", "webrtc", "energy"]


class VadUnavailable(RuntimeError):
    pass


class VadBackend(Protocol):
    name: str
    sample_rate: int
    frame_samples: int

    def __call__(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


# --- Silero -------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_silero_model(cache_dir: str | Path, explicit: str | None = None, allow_download: bool = True) -> Path:
    """Return a verified Silero ONNX path, downloading the pinned release once if allowed."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise VadUnavailable(f"vad.silero_model_path does not exist: {path}")
        return path
    path = Path(cache_dir) / "models" / f"silero_vad_{SILERO_VERSION}" / "silero_vad.onnx"
    if not path.is_file():
        if not allow_download:
            raise VadUnavailable(f"Silero model missing at {path} and vad.allow_model_download=false")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(SILERO_URL, tmp)  # noqa: S310 - pinned https URL
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise VadUnavailable(f"Silero download failed: {exc}") from exc
        tmp.replace(path)
    digest = sha256_file(path)
    if digest != SILERO_SHA256:
        raise VadUnavailable(f"Silero model checksum mismatch at {path} (got {digest[:12]}...)")
    return path


class SileroVad:
    name = "silero"

    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        if sample_rate not in (8000, 16000):
            raise VadUnavailable("Silero VAD supports 8 kHz and 16 kHz only")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise VadUnavailable("onnxruntime not installed (pip install onnxruntime)") from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1   # tiny model: threading overhead dominates
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self.sample_rate = sample_rate
        self.frame_samples = 512 if sample_rate == 16000 else 256
        self._ctx_samples = 64 if sample_rate == 16000 else 32
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._ctx_samples), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -self._ctx_samples:]
        return float(out[0, 0])


# --- WebRTC -------------------------------------------------------------------------------------
class WebRtcVad:
    name = "webrtc"

    def __init__(self, threshold: float, sample_rate: int = 16000, frame_ms: int = 30) -> None:
        try:
            import webrtcvad
        except ImportError as exc:
            raise VadUnavailable("webrtcvad not installed (pip install webrtcvad-wheels)") from exc
        if sample_rate not in (8000, 16000, 32000, 48000) or frame_ms not in (10, 20, 30):
            raise VadUnavailable("WebRTC VAD needs 8/16/32/48 kHz and 10/20/30 ms frames")
        # threshold 0..1 -> aggressiveness 0..3 (higher threshold = stricter)
        self.aggressiveness = min(3, int(threshold * 4))
        self._vad = webrtcvad.Vad(self.aggressiveness)
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000

    def reset(self) -> None:
        pass  # stateless per frame

    def __call__(self, frame: np.ndarray) -> float:
        return 1.0 if self._vad.is_speech(float_to_pcm16(frame), self.sample_rate) else 0.0


# --- Energy -------------------------------------------------------------------------------------
class EnergyVad:
    """Speech probability from frame energy relative to a tracked noise floor."""

    name = "energy"

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, margin_db: float = 12.0,
                 slope_db: float = 3.0, floor_db: float = -60.0) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.margin_db, self.slope_db, self._init_floor = margin_db, slope_db, floor_db
        self.reset()

    def reset(self) -> None:
        self.noise_floor_db = self._init_floor

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        db = 20 * math.log10(rms + 1e-10)
        # Floor follows quiet frames quickly and loud frames very slowly.
        rate = 0.2 if db < self.noise_floor_db else 0.002
        self.noise_floor_db += rate * (db - self.noise_floor_db)
        z = (db - (self.noise_floor_db + self.margin_db)) / self.slope_db
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


# --- Factory ------------------------------------------------------------------------------------
def _build(name: str, cfg: VadConfig, sample_rate: int, frame_ms: int, cache_dir: str) -> VadBackend:
    if name == "silero":
        path = ensure_silero_model(cache_dir, cfg.silero_model_path, cfg.allow_model_download)
        return SileroVad(path, sample_rate)
    if name == "webrtc":
        return WebRtcVad(cfg.threshold, sample_rate, frame_ms)
    if name == "energy":
        return EnergyVad(sample_rate, frame_ms)
    raise VadUnavailable(f"unknown VAD backend {name!r}")


def create_vad(
    cfg: VadConfig,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    cache_dir: str = "cache",
    fallbacks: FallbackLog | None = None,
    allow_fallback: bool = True,
) -> VadBackend:
    """Build the configured backend, falling back down silero -> webrtc -> energy (logged)."""
    order = [cfg.backend] + [b for b in CHAIN[CHAIN.index(cfg.backend) + 1:]]
    errors: list[str] = []
    for i, name in enumerate(order):
        try:
            backend = _build(name, cfg, sample_rate, frame_ms, cache_dir)
        except VadUnavailable as exc:
            errors.append(f"{name}: {exc}")
            if not allow_fallback:
                raise
            continue
        if i > 0 and fallbacks is not None:
            fallbacks.add("vad", cfg.backend, name, " | ".join(errors))
        return backend
    raise VadUnavailable("no VAD backend available: " + " | ".join(errors))
