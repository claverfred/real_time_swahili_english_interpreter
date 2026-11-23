"""Streaming speech segmentation on top of any frame-level VAD backend.

State machine::

    SILENCE --(min_speech_ms of consecutive speech)--> SPEECH   emits "start" (+ speech_pad_ms pre-roll)
    SPEECH  --(every frame)--------------------------> SPEECH   emits "audio" (raw stream for live ASR)
    SPEECH  --(min_silence_ms of non-speech)---------> SILENCE  emits "end"   (trimmed to last speech + hangover_ms)
    SPEECH  --(max_utterance_s reached)--------------> SPEECH   emits "end" (reason max_duration), new segment
    any     --flush()--------------------------------> SILENCE  emits "end" (reason flush) if speaking

While in SPEECH a lower exit threshold (threshold - 0.15) is used, as in Silero's reference
implementation, so brief dips do not cut words. Timestamps are counted in samples from stream
start, so results do not depend on how audio is chunked.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..audio.buffer import FrameChunker
from ..config.schema import VadConfig
from ..pipeline.events import EventRecorder, EventType
from .backends import VadBackend


@dataclass
class SpeechSegment:
    segment_id: int
    start_s: float
    end_s: float
    audio: np.ndarray = field(repr=False)
    reason: Literal["silence", "max_duration", "flush"]
    mean_prob: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, object]:
        return {"segment_id": self.segment_id, "start_s": round(self.start_s, 3),
                "end_s": round(self.end_s, 3), "duration_s": round(self.duration_s, 3),
                "reason": self.reason, "mean_prob": round(self.mean_prob, 4)}


@dataclass
class VadEvent:
    kind: Literal["start", "audio", "end"]
    segment_id: int
    t_s: float
    audio: np.ndarray | None = field(default=None, repr=False)
    segment: SpeechSegment | None = None


_Frame = tuple[int, np.ndarray, float]   # (frame index, samples, speech prob)


class VadSegmenter:
    def __init__(self, backend: VadBackend, cfg: VadConfig, recorder: EventRecorder | None = None) -> None:
        self.backend = backend
        self.cfg = cfg
        self.recorder = recorder
        self.sample_rate = backend.sample_rate
        self.frame_samples = backend.frame_samples
        self.frame_ms = self.frame_samples * 1000.0 / self.sample_rate
        f = self.frame_ms
        self._pad_frames = math.ceil(cfg.speech_pad_ms / f)
        self._min_speech = max(1, math.ceil(cfg.min_speech_ms / f))
        self._min_silence = max(1, math.ceil(cfg.min_silence_ms / f))
        self._hangover = math.ceil(cfg.hangover_ms / f)
        self._max_frames = max(1, int(cfg.max_utterance_s * 1000 / f))
        self.threshold = cfg.threshold
        self.exit_threshold = max(cfg.threshold - 0.15, 0.01)
        self._chunker = FrameChunker(self.frame_samples)
        self.reset()

    # -- public API ---------------------------------------------------------------------------
    def reset(self) -> None:
        self.backend.reset()
        self._chunker.reset()
        self._frame_idx = 0
        self._next_id = 0
        self._speaking = False
        self._segment_id = -1
        self._preroll: deque[_Frame] = deque(maxlen=max(self._pad_frames, 1) if self._pad_frames else 0)
        self._candidates: list[_Frame] = []
        self._seg: list[_Frame] = []
        self._last_speech = -1
        self._silence_run = 0

    def discard(self) -> None:
        """Drop speech in progress (cancel) without restarting the timeline or segment ids."""
        self.backend.reset()
        self._speaking = False
        self._preroll.clear()
        self._candidates = []
        self._seg = []
        self._last_speech = -1
        self._silence_run = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def process(self, chunk: np.ndarray) -> list[VadEvent]:
        if self.recorder is not None:
            self.recorder.emit(EventType.AUDIO_FRAME_RECEIVED, samples=int(len(chunk)))
        out: list[VadEvent] = []
        for frame in self._chunker.push(chunk):
            self._step(frame, out)
        return _coalesce(out)

    def flush(self) -> list[VadEvent]:
        out: list[VadEvent] = []
        last = self._chunker.flush()
        if last is not None:
            self._step(last, out)
        if self._speaking and self._seg:
            self._close("flush", out)
        self._speaking = False
        self._candidates.clear()   # shorter than min_speech_ms: not speech
        return _coalesce(out)

    # -- internals ----------------------------------------------------------------------------
    def _t(self, frame_idx: int) -> float:
        return frame_idx * self.frame_samples / self.sample_rate

    def _step(self, frame: np.ndarray, out: list[VadEvent]) -> None:
        idx = self._frame_idx
        self._frame_idx += 1
        prob = float(self.backend(frame))
        item = (idx, frame, prob)

        if not self._speaking:
            if prob >= self.threshold:
                self._candidates.append(item)
                if len(self._candidates) >= self._min_speech:
                    self._open(out)
            else:
                if self._pad_frames:
                    self._preroll.extend(self._candidates)
                    self._preroll.append(item)
                self._candidates.clear()
            return

        if not self._seg:   # continuation after a max_duration split
            self._emit_start(idx, out, frame)
        else:
            out.append(VadEvent("audio", self._segment_id, self._t(idx), frame))
        self._seg.append(item)
        if prob >= self.exit_threshold:
            self._silence_run = 0
            self._last_speech = len(self._seg) - 1
        else:
            self._silence_run += 1

        if self._silence_run >= self._min_silence:
            self._close("silence", out)
        elif len(self._seg) >= self._max_frames:
            self._close("max_duration", out)
            self._speaking = True      # keep speaking; next frame opens a new segment

    def _emit_start(self, first_idx: int, out: list[VadEvent], audio: np.ndarray) -> None:
        self._segment_id = self._next_id
        self._next_id += 1
        t = self._t(first_idx)
        out.append(VadEvent("start", self._segment_id, t, audio))
        if self.recorder is not None:
            self.recorder.emit(EventType.VAD_SPEECH_START, segment_id=self._segment_id, start_s=round(t, 3))

    def _open(self, out: list[VadEvent]) -> None:
        frames = list(self._preroll) + self._candidates
        self._preroll.clear()
        self._candidates = []
        self._seg = frames
        self._last_speech = len(frames) - 1
        self._silence_run = 0
        self._speaking = True
        self._emit_start(frames[0][0], out, np.concatenate([f[1] for f in frames]))

    def _close(self, reason: Literal["silence", "max_duration", "flush"], out: list[VadEvent]) -> None:
        if reason == "max_duration":
            keep = len(self._seg)
        else:
            # max(1, ...) covers a post-split continuation that never contained speech.
            keep = max(1, min(len(self._seg), self._last_speech + 1 + self._hangover))
        kept, tail = self._seg[:keep], self._seg[keep:]
        start_s = self._t(kept[0][0])
        end_s = self._t(kept[-1][0] + 1)
        seg = SpeechSegment(
            segment_id=self._segment_id,
            start_s=start_s,
            end_s=end_s,
            audio=np.concatenate([f[1] for f in kept]),
            reason=reason,
            mean_prob=float(np.mean([f[2] for f in kept])),
        )
        out.append(VadEvent("end", self._segment_id, end_s, segment=seg))
        if self.recorder is not None:
            self.recorder.emit(EventType.VAD_SPEECH_END, segment_id=self._segment_id,
                               start_s=round(start_s, 3), end_s=round(end_s, 3),
                               duration_s=round(seg.duration_s, 3), reason=reason)
        self._seg = []
        self._speaking = False
        self._silence_run = 0
        self._last_speech = -1
        self._preroll.clear()
        if self._pad_frames:
            self._preroll.extend(tail)


def _coalesce(events: list[VadEvent]) -> list[VadEvent]:
    """Merge consecutive "audio" events of the same segment into one array."""
    merged: list[VadEvent] = []
    pending: list[np.ndarray] = []
    for ev in events:
        if ev.kind == "audio" and merged and merged[-1].kind == "audio" and merged[-1].segment_id == ev.segment_id:
            pending.append(ev.audio)  # type: ignore[arg-type]
            continue
        if pending:
            merged[-1].audio = np.concatenate([merged[-1].audio, *pending])  # type: ignore[list-item]
            pending = []
        merged.append(ev)
    if pending:
        merged[-1].audio = np.concatenate([merged[-1].audio, *pending])  # type: ignore[list-item]
    return merged


def segment_audio(audio: np.ndarray, segmenter: VadSegmenter, chunk_s: float = 0.5) -> list[SpeechSegment]:
    """Run a whole array through a segmenter (batch mode) and return the closed segments."""
    step = max(1, int(chunk_s * segmenter.sample_rate))
    events: list[VadEvent] = []
    for i in range(0, len(audio), step):
        events.extend(segmenter.process(audio[i:i + step]))
    events.extend(segmenter.flush())
    return [e.segment for e in events if e.kind == "end" and e.segment is not None]
