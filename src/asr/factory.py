"""Build the configured ASR model, walking ``asr.fallback_models`` on failure (always logged)."""

from __future__ import annotations

import gc

from ..config.schema import InterpreterConfig
from ..platform.resolve import FallbackRefused
from ..utils.manifest import FallbackLog
from .faster_whisper_backend import AsrError, FasterWhisperAsr


def create_asr(cfg: InterpreterConfig, fallbacks: FallbackLog | None = None,
               local_files_only: bool = False) -> FasterWhisperAsr:
    """``cfg`` must already be runtime-resolved (concrete device and compute type)."""
    if cfg.asr.backend != "faster_whisper":
        raise AsrError(f"ASR backend {cfg.asr.backend!r} is not implemented yet")
    env = cfg.environment
    candidates = [cfg.asr.model, *[m for m in cfg.asr.fallback_models if m != cfg.asr.model]]
    errors: list[str] = []
    for i, model in enumerate(candidates):
        asr = FasterWhisperAsr(
            cfg.asr, model=model, device=env.device, compute_type=cfg.asr.compute_type,
            cpu_threads=env.cpu_threads, device_index=env.gpu_index, cache_dir=cfg.storage.cache_dir,
            local_files_only=local_files_only, sample_rate=cfg.audio.sample_rate,
        )
        try:
            asr.load()
        except AsrError as exc:
            errors.append(f"{model}: {exc}")
            gc.collect()
            if cfg.fallback.policy == "fail":
                raise FallbackRefused(f"ASR model {model} failed and fallback.policy=fail: {exc}") from exc
            continue
        if i > 0 and fallbacks is not None:
            fallbacks.add("asr", cfg.asr.model, model, " | ".join(errors))
        return asr
    raise AsrError("no ASR model could be loaded: " + " | ".join(errors))
