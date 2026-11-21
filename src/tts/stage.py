"""MT finals -> English speech chunks, with spec event tracing.

Only final translations are spoken: audio cannot be revised once played, so provisional
(partial) translations are never synthesized. Events per unit:
``tts_chunk_requested`` -> ``tts_first_audio`` (first chunk of the unit) -> ``tts_audio``
(payload, not persisted) -> ``tts_chunk_complete`` ... -> ``tts_complete``.
``first_audio_ms`` is measured from the moment the unit reached TTS.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..mt.stage import MtUpdate
from ..pipeline.events import EventRecorder, EventType
from .synthesizer import SynthChunk, Synthesizer


@dataclass
class TtsUpdate:
    unit_id: int | None
    segment_id: int | None
    chunk: SynthChunk
    is_first: bool
    is_last: bool
    first_audio_ms: float
    extra: dict[str, object] = field(default_factory=dict)


class TtsStage:
    def __init__(self, synth: Synthesizer, recorder: EventRecorder | None = None) -> None:
        self.synth, self.recorder = synth, recorder
        self.chunks_done = 0
        self.session_first_audio_t: float | None = None

    def _emit(self, etype: EventType, **kw) -> None:
        if self.recorder is not None:
            self.recorder.emit(etype, **kw)

    def speak(self, text: str, *, unit_id: int | None = None, segment_id: int | None = None) -> Iterator[TtsUpdate]:
        if not text.strip():
            return
        t0 = time.perf_counter()
        pieces = self.synth.chunks(text)
        first_ms = 0.0
        for i, piece in enumerate(pieces):
            self._emit(EventType.TTS_CHUNK_REQUESTED, segment_id=segment_id, text=piece, unit_id=unit_id,
                       chunk=i, chars=len(piece))
            before = len(self.synth.switches)
            chunk = self.synth.synthesize_chunk(piece, i)
            for sw in self.synth.switches[before:]:
                self._emit(EventType.FALLBACK, segment_id=segment_id, stage="tts", **sw)
            if i == 0:
                first_ms = (time.perf_counter() - t0) * 1000
                if self.session_first_audio_t is None:
                    self.session_first_audio_t = time.perf_counter()
                self._emit(EventType.TTS_FIRST_AUDIO, segment_id=segment_id, unit_id=unit_id,
                           first_audio_ms=round(first_ms, 2), backend=chunk.backend)
            self._emit(EventType.TTS_AUDIO, segment_id=segment_id, audio=chunk.audio, unit_id=unit_id, chunk=i,
                       sample_rate=chunk.sample_rate, duration_s=round(chunk.duration_s, 3))
            self._emit(EventType.TTS_CHUNK_COMPLETE, segment_id=segment_id, unit_id=unit_id, chunk=i,
                       synth_ms=chunk.synth_ms, duration_s=round(chunk.duration_s, 3), rtf=round(chunk.rtf, 4),
                       backend=chunk.backend, warnings=chunk.warnings)
            self.chunks_done += 1
            yield TtsUpdate(unit_id, segment_id, chunk, i == 0, i == len(pieces) - 1, round(first_ms, 2))
        self._emit(EventType.TTS_COMPLETE, segment_id=segment_id, unit_id=unit_id, chunks=len(pieces),
                   total_ms=round((time.perf_counter() - t0) * 1000, 2))

    def on_mt_update(self, up: MtUpdate) -> list[TtsUpdate]:
        if up.kind != "final" or up.result is None or not up.result.text:
            return []
        unit_id = up.unit.unit_id if up.unit else None
        return list(self.speak(up.result.text, unit_id=unit_id, segment_id=up.segment_id))
