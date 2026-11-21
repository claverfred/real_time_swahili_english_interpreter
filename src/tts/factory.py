"""Build the synthesizer from ``tts.backend`` + ``tts.fallback_backends`` (all fallbacks logged).

The first backend that loads becomes active. The ones after it stay as lazy runtime fallbacks.
Cloud backends are only reachable here if the config validator allowed them (allow_cloud,
data class).
"""

from __future__ import annotations

import gc

from ..config.schema import CLOUD_TTS_BACKENDS, InterpreterConfig
from ..platform.resolve import FallbackRefused
from ..utils.manifest import FallbackLog
from .backends import TtsBackend, TtsError, make_backend
from .synthesizer import Synthesizer
from .text import PronunciationDict


def _voice_for(cfg: InterpreterConfig, name: str) -> str:
    if name == cfg.tts.backend:
        return cfg.tts.voice
    try:
        return cfg.tts.voices[name]
    except KeyError as exc:
        raise TtsError(f"tts.voices has no voice for fallback backend {name!r}") from exc


def create_synthesizer(cfg: InterpreterConfig, fallbacks: FallbackLog | None = None,
                       backends: list[TtsBackend] | None = None) -> Synthesizer:
    tts = cfg.tts
    names = [tts.backend, *[b for b in tts.fallback_backends if b != tts.backend]]
    if backends is None:
        backends = []
        for name in names:
            if name in CLOUD_TTS_BACKENDS and (not cfg.fallback.allow_cloud or
                                               cfg.privacy.data_class in {"restricted", "private"}):
                raise TtsError(f"cloud TTS backend {name!r} is not permitted by this configuration")
            backends.append(make_backend(
                name, _voice_for(cfg, name), cache_root=cfg.storage.cache_dir, speaker_id=tts.speaker_id,
                speaking_rate=tts.speaking_rate, allow_download=tts.allow_model_download,
                revision=tts.revision if name == tts.backend else "main", timeout_s=cfg.streaming.timeout_s))
    pron = PronunciationDict.load(tts.pronunciation_dict) if tts.pronunciation_dict else None

    errors: list[str] = []
    for i, b in enumerate(backends):
        try:
            b.load()
        except TtsError as exc:
            errors.append(f"{b.name}: {exc}")
            gc.collect()
            if cfg.fallback.policy == "fail":
                raise FallbackRefused(f"TTS backend {b.name} failed and fallback.policy=fail: {exc}") from exc
            continue
        if i > 0 and fallbacks is not None:
            fallbacks.add("tts", tts.backend, b.name, " | ".join(errors))

        def on_fallback(src: str, dst: str, reason: str) -> None:
            if fallbacks is not None:
                fallbacks.add("tts", src, dst, f"runtime failure: {reason}")

        return Synthesizer(backends[i:], tts, pron, retry_count=cfg.streaming.retry_count, on_fallback=on_fallback)
    raise TtsError("no TTS backend could be loaded: " + " | ".join(errors))
