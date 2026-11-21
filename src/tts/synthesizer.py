"""Synthesizer: chunking, spoken-form normalization, backend calls with retries and runtime
fallback, and consistent post-processing.

A backend that fails at synthesis time (e.g. the network drops for edge) is retried
``streaming.retry_count`` times, then the next configured backend takes over for the rest of
the session. The switch is recorded (it changes the voice, so it is never silent).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from ..config.schema import TtsConfig
from .audio import postprocess, validate_output
from .backends import TtsBackend, TtsError
from .text import PronunciationDict, chunk_text, normalize_for_tts


@dataclass
class SynthChunk:
    index: int
    text: str                 # English text of this chunk
    spoken: str               # normalized text sent to the backend
    audio: np.ndarray         # float32 mono at Synthesizer.sample_rate, without the trailing pause
    sample_rate: int
    synth_ms: float
    backend: str
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def rtf(self) -> float:
        return self.synth_ms / 1000 / self.duration_s if self.duration_s else float("inf")


@dataclass
class SynthResult:
    text: str
    chunks: list[SynthChunk]
    audio: np.ndarray
    sample_rate: int
    first_audio_ms: float
    total_ms: float

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def rtf(self) -> float:
        return self.total_ms / 1000 / self.duration_s if self.duration_s else float("inf")


class Synthesizer:
    def __init__(self, backends: list[TtsBackend] | TtsBackend, cfg: TtsConfig,
                 pron: PronunciationDict | None = None, retry_count: int = 1,
                 on_fallback: Callable[[str, str, str], None] | None = None,
                 loader: Callable[[TtsBackend], None] | None = None) -> None:
        self.backends = backends if isinstance(backends, list) else [backends]
        self.cfg, self.pron, self.retry_count = cfg, pron, retry_count
        self.on_fallback, self._loader = on_fallback, loader or (lambda b: b.load())
        self._active = 0
        self.sample_rate = cfg.sample_rate
        self.switches: list[dict[str, str]] = []

    @property
    def backend(self) -> TtsBackend:
        return self.backends[self._active]

    @property
    def load_info(self) -> dict[str, object]:
        return self.backend.load_info

    def _call_backend(self, spoken: str) -> tuple[np.ndarray, int, str]:
        while True:
            b = self.backend
            last: Exception | None = None
            for _ in range(self.retry_count + 1):
                try:
                    return b.synthesize(spoken), b.sample_rate, b.name
                except Exception as exc:  # backend-specific errors all become a retry / fallback
                    last = exc
            if self._active + 1 >= len(self.backends):
                raise TtsError(f"TTS backend {b.name} failed and no fallback remains: {last}") from last
            nxt = self.backends[self._active + 1]
            try:
                if not nxt.load_info:
                    self._loader(nxt)
            except Exception as exc:
                self._active += 1  # skip a fallback that cannot load
                last = exc
                continue
            self.switches.append({"from": b.name, "to": nxt.name, "reason": str(last)})
            if self.on_fallback:
                self.on_fallback(b.name, nxt.name, str(last))
            self._active += 1

    def synthesize_chunk(self, text: str, index: int = 0) -> SynthChunk:
        spoken = normalize_for_tts(text, pron=self.pron, spell_acronyms=self.cfg.spell_acronyms)
        t0 = time.perf_counter()
        raw, sr, name = self._call_backend(spoken)
        audio = postprocess(raw, sr, self.sample_rate, trim=self.cfg.trim_silence, fade_ms=self.cfg.fade_ms,
                            target_rms_dbfs=self.cfg.target_rms_dbfs)
        ms = (time.perf_counter() - t0) * 1000
        return SynthChunk(index, text, spoken, audio, self.sample_rate, round(ms, 2), name,
                          validate_output(audio, self.sample_rate, spoken))

    def chunks(self, text: str) -> list[str]:
        return chunk_text(text, self.cfg.chunk_max_chars, self.cfg.first_chunk_max_chars)

    def stream(self, text: str) -> Iterator[SynthChunk]:
        for i, chunk in enumerate(self.chunks(text)):
            yield self.synthesize_chunk(chunk, i)

    def pause(self) -> np.ndarray:
        return np.zeros(int(self.sample_rate * self.cfg.sentence_pause_ms / 1000), np.float32)

    def synthesize(self, text: str) -> SynthResult:
        t0 = time.perf_counter()
        chunks: list[SynthChunk] = []
        first_ms = 0.0
        for c in self.stream(text):
            if not chunks:
                first_ms = (time.perf_counter() - t0) * 1000
            chunks.append(c)
        pieces: list[np.ndarray] = []
        for i, c in enumerate(chunks):
            pieces.append(c.audio)
            if i < len(chunks) - 1:
                pieces.append(self.pause())
        audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        return SynthResult(text, chunks, audio, self.sample_rate, round(first_ms, 2),
                           round((time.perf_counter() - t0) * 1000, 2))

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.backend.synthesize("Hello.")
        return (time.perf_counter() - t0) * 1000
