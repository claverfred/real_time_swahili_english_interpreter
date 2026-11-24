"""faster-whisper (CTranslate2) ASR backend.

Download, model load, and inference are timed separately so cold start, warm start, and
inference latency are never conflated. The resolved Hugging Face snapshot commit is recorded
for reproducibility.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config.schema import AsrConfig
from .filters import filter_segments
from .types import AsrResult, AsrSegment, AsrWord

MIN_AUDIO_S = 0.1   # below this nothing meaningful can be decoded; return empty, do not guess


class AsrError(RuntimeError):
    pass


class AsrOutOfMemory(AsrError):
    pass


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


class FasterWhisperAsr:
    name = "faster_whisper"

    def __init__(self, cfg: AsrConfig, *, model: str | None = None, device: str = "cpu",
                 compute_type: str = "int8", cpu_threads: int = 0, device_index: int = 0,
                 cache_dir: str | Path | None = None, local_files_only: bool = False,
                 sample_rate: int = 16000) -> None:
        self.cfg = cfg
        self.model_name = model or cfg.model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.device_index = device_index
        self.cache_dir = Path(cache_dir) / "huggingface" / "hub" if cache_dir else None
        self.local_files_only = local_files_only
        self.sample_rate = sample_rate
        self._model: Any = None
        self.load_info: dict[str, object] = {}

    # -- lifecycle ----------------------------------------------------------------------------
    def load(self) -> dict[str, object]:
        try:
            from faster_whisper import WhisperModel
            from faster_whisper.utils import download_model
        except ImportError as exc:
            raise AsrError('faster-whisper not installed (pip install -e ".[asr]")') from exc

        from .convert import AsrConvertError, ensure_ct2_whisper, is_builtin

        revision = None if self.cfg.revision in ("", "main") else self.cfg.revision
        t0 = time.perf_counter()
        convert_info: dict[str, object] = {}
        try:
            if is_builtin(self.model_name):
                path = download_model(self.model_name, local_files_only=self.local_files_only,
                                      cache_dir=str(self.cache_dir) if self.cache_dir else None, revision=revision)
            else:   # any Hugging Face Whisper repo (e.g. a Swahili fine-tune): convert once if needed
                cache_root = self.cache_dir.parent.parent if self.cache_dir else Path("cache")
                p, convert_info = ensure_ct2_whisper(self.model_name, revision=revision or "main",
                                                     compute_type=self.compute_type, cache_root=cache_root,
                                                     local_files_only=self.local_files_only)
                path = str(p)
        except (AsrConvertError, Exception) as exc:
            raise AsrError(f"could not obtain model {self.model_name!r}: {exc}") from exc
        t1 = time.perf_counter()
        try:
            self._model = WhisperModel(path, device=self.device, device_index=self.device_index,
                                       compute_type=self.compute_type, cpu_threads=self.cpu_threads)
        except Exception as exc:
            if _is_oom(exc):
                raise AsrOutOfMemory(f"out of memory loading {self.model_name} ({self.compute_type})") from exc
            raise AsrError(f"could not load {self.model_name} on {self.device}/{self.compute_type}: {exc}") from exc
        t2 = time.perf_counter()
        p = Path(path)
        self.load_info = {
            "model": self.model_name,
            "path": str(p),
            "resolved_revision": convert_info.get("resolved_revision")
            or (p.name if p.parent.name == "snapshots" else self.cfg.revision),
            "conversion": convert_info or None,
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "download_or_cache_ms": round((t1 - t0) * 1000, 1),
            "load_ms": round((t2 - t1) * 1000, 1),
        }
        return self.load_info

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None

    def warmup(self) -> float:
        """One throwaway decode so first-request latency is measured separately (ms)."""
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(self.sample_rate) * 0.01).astype(np.float32)
        return self.transcribe(noise, fast=True).inference_ms

    # -- inference ----------------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, *, fast: bool = False, initial_prompt: str | None = None,
                   previous_text: str = "", word_timestamps: bool | None = None) -> AsrResult:
        """Decode one utterance. ``fast`` = greedy, single temperature, no timestamps (partials)."""
        if self._model is None:
            raise AsrError("model not loaded; call load() first")
        duration = len(audio) / self.sample_rate
        if duration < MIN_AUDIO_S:
            return AsrResult("", self.cfg.language, True, duration, 0.0, self.model_name)
        wt = self.cfg.word_timestamps if word_timestamps is None else word_timestamps
        kwargs: dict[str, Any] = dict(
            language=self.cfg.language,
            task=self.cfg.task,
            beam_size=1 if fast else self.cfg.beam_size,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
            word_timestamps=wt and not fast,
            initial_prompt=initial_prompt,
            vad_filter=False,              # segmentation is done by our VAD stage
        )
        if fast:
            kwargs.update(temperature=0.0, without_timestamps=True)
        else:
            kwargs.update(temperature=tuple(self.cfg.temperatures or [0.0]))
        t0 = time.perf_counter()
        try:
            seg_iter, info = self._model.transcribe(audio.astype(np.float32, copy=False), **kwargs)
            raw = list(seg_iter)       # the generator does the actual decoding
        except Exception as exc:
            if _is_oom(exc):
                raise AsrOutOfMemory(f"out of memory decoding {duration:.1f} s audio") from exc
            raise AsrError(f"decode failed: {exc}") from exc
        ms = (time.perf_counter() - t0) * 1000

        segments = [
            AsrSegment(
                start=float(s.start), end=float(s.end), text=s.text,
                avg_logprob=float(s.avg_logprob), no_speech_prob=float(s.no_speech_prob),
                compression_ratio=float(s.compression_ratio), temperature=float(s.temperature or 0.0),
                words=[AsrWord(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                       for w in (s.words or [])],
            )
            for s in raw
        ]
        filter_segments(segments, self.cfg, previous_text)
        text = " ".join(s.text for s in segments if s.rejected is None).strip()
        return AsrResult(text=text, language=info.language, language_forced=True,
                         duration_s=duration, inference_ms=round(ms, 2), model=self.model_name,
                         segments=segments)
