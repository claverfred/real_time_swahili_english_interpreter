"""Streaming ASR on top of VAD segments: partial hypotheses, stable prefixes, final segments.

Per VAD segment:
  * while speech continues, the growing segment audio is re-decoded (greedy) every
    ``asr.chunk_s`` seconds of new audio -> ``partial`` updates;
  * a word prefix is *committed* once ``asr.stable_prefix_agreement`` consecutive hypotheses agree
    on it (LocalAgreement-n). Committed text never shrinks; later disagreement is counted as a
    revision, not silently rewritten;
  * on VAD end the full segment is decoded with the configured beam -> ``final`` update. The final
    is authoritative and is what downstream MT consumes (``mt.translate_partials`` is off by default).

Partial decoding adapts to hardware: the next partial waits for at least
``max(chunk_s, 1.5 x last partial decode time)`` of new audio, so a slow CPU degrades to fewer
partials instead of an ever-growing backlog.

After a ``max_duration`` split, the last ``asr.overlap_s`` of the previous segment is prepended to
the next one, and words duplicated across the boundary are removed with :func:`merge_overlap`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from ..config.schema import AsrConfig
from ..pipeline.events import EventRecorder, EventType
from ..vad.segmenter import VadEvent
from .corrections import AsrCorrections
from .filters import common_prefix, merge_overlap
from .types import AsrResult


class Transcriber(Protocol):
    model_name: str

    def transcribe(self, audio: np.ndarray, *, fast: bool = False, initial_prompt: str | None = None,
                   previous_text: str = "", word_timestamps: bool | None = None) -> AsrResult: ...


@dataclass
class AsrUpdate:
    kind: Literal["partial", "final"]
    segment_id: int
    text: str
    stable: str = ""
    unstable: str = ""
    latency_ms: float = 0.0
    audio_s: float = 0.0
    result: AsrResult | None = None
    revisions: int = 0
    start_s: float = 0.0
    end_s: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


class StreamingTranscriber:
    def __init__(self, asr: Transcriber, cfg: AsrConfig, sample_rate: int = 16000,
                 recorder: EventRecorder | None = None, partials: bool = True,
                 corrections: AsrCorrections | None = None) -> None:
        self.asr = asr
        self.cfg = cfg
        self.sr = sample_rate
        self.recorder = recorder
        self.partials = partials
        if corrections is None:
            corrections = AsrCorrections.load(cfg.corrections_path) if cfg.corrections_path else AsrCorrections()
        self.corrections = corrections
        self.corrections_applied = 0
        self.reset()

    def reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._new_samples = 0
        self._interval_s = self.cfg.chunk_s
        self._hyps: deque[list[str]] = deque(maxlen=self.cfg.stable_prefix_agreement)
        self._committed: list[str] = []
        self._revisions = 0
        self._segment_id = -1
        self._carry: np.ndarray | None = None      # overlap audio after a max_duration split
        self._prefix: np.ndarray | None = None     # carry consumed by the current segment
        self._carry_text = ""                       # final text of the split-off previous part
        self._previous_final = ""
        self.partial_count = 0
        self.final_count = 0

    def discard(self) -> None:
        """Drop the segment in progress and any split-overlap carry (cancel); keep counters."""
        self._chunks, self._new_samples = [], 0
        self._hyps.clear()
        self._committed = []
        self._carry = self._prefix = None
        self._carry_text = ""
        self._segment_id = -1

    # -- driving ------------------------------------------------------------------------------
    def on_vad_event(self, ev: VadEvent) -> list[AsrUpdate]:
        if ev.kind == "start":
            self._begin(ev)
            return []
        if ev.kind == "audio":
            if ev.segment_id != self._segment_id or ev.audio is None:
                return []
            self._chunks.append(ev.audio)
            self._new_samples += len(ev.audio)
            if self.partials and self._new_samples >= self._interval_s * self.sr:
                upd = self._partial()
                return [upd] if upd else []
            return []
        return [self._final(ev)]

    def _begin(self, ev: VadEvent) -> None:
        self._segment_id = ev.segment_id
        self._chunks = []
        self._prefix = self._carry          # overlap audio from a max_duration split, if any
        self._carry = None
        if self._prefix is not None:
            self._chunks.append(self._prefix)
        if ev.audio is not None:
            self._chunks.append(ev.audio)
        self._new_samples = sum(len(c) for c in self._chunks)
        self._hyps.clear()
        self._committed = []
        self._revisions = 0

    def _prompt(self) -> str | None:
        if not self.cfg.condition_on_previous_text:
            return None
        return self._previous_final[-200:] or None

    # -- partials -----------------------------------------------------------------------------
    def _partial(self) -> AsrUpdate | None:
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._new_samples = 0
        res = self.asr.transcribe(audio, fast=True, initial_prompt=self._prompt(),
                                  previous_text=self._previous_final)
        self._interval_s = max(self.cfg.chunk_s, 1.5 * res.inference_ms / 1000.0)
        fixed, _ = self.corrections.apply(res.text)
        hyp = merge_overlap(self._carry_text, fixed).split() if self._carry_text else fixed.split()
        self._hyps.append(hyp)
        if len(self._hyps) == self._hyps.maxlen:
            agreed = min(common_prefix(self._hyps[0], h) for h in self._hyps)
            if agreed > len(self._committed):
                self._committed = hyp[:agreed]
            elif common_prefix(self._committed, hyp) < len(self._committed):
                self._revisions += 1
        stable = " ".join(self._committed)
        unstable = " ".join(hyp[common_prefix(self._committed, hyp):])
        text = f"{stable} {unstable}".strip()
        self.partial_count += 1
        upd = AsrUpdate("partial", self._segment_id, text, stable, unstable, res.inference_ms,
                        len(audio) / self.sr, res, self._revisions)
        if self.recorder is not None:
            self.recorder.emit(EventType.ASR_PARTIAL, segment_id=self._segment_id, text=text, stable=stable,
                               unstable=unstable, latency_ms=res.inference_ms, audio_s=round(upd.audio_s, 3))
        return upd

    # -- finals -------------------------------------------------------------------------------
    def _final(self, ev: VadEvent) -> AsrUpdate:
        seg = ev.segment
        assert seg is not None, "end events carry the closed segment"
        prefix = self._prefix if seg.segment_id == self._segment_id else None
        audio = np.concatenate([prefix, seg.audio]) if prefix is not None else seg.audio
        res = self.asr.transcribe(audio, initial_prompt=self._prompt(), previous_text=self._previous_final)
        fixed, n_fixed = self.corrections.apply(res.text)
        self.corrections_applied += n_fixed
        text = merge_overlap(self._carry_text, fixed) if self._carry_text else fixed
        final_words = text.split()
        if self._committed and common_prefix(self._committed, final_words) < len(self._committed):
            self._revisions += 1

        if seg.reason == "max_duration" and self.cfg.overlap_s > 0:
            n = int(self.cfg.overlap_s * self.sr)
            self._carry = seg.audio[-n:].copy()
            self._carry_text = text
        else:
            self._carry_text = ""
        if text:
            self._previous_final = text
        self.final_count += 1
        self._chunks = []
        self._prefix = None
        rejected = [s.rejected for s in res.rejected_segments]
        upd = AsrUpdate("final", seg.segment_id, text, stable=text, latency_ms=res.inference_ms,
                        audio_s=len(audio) / self.sr, result=res, revisions=self._revisions,
                        start_s=seg.start_s, end_s=seg.end_s,
                        extra={"rejected": rejected, "uncertain": res.uncertain, "vad_reason": seg.reason,
                               "corrections": n_fixed, "original": res.text})
        if self.recorder is not None:
            fix_info = {"corrections": n_fixed, "original": res.text} if n_fixed else {}
            self.recorder.emit(EventType.ASR_FINAL, segment_id=seg.segment_id, text=text, **fix_info,
                               latency_ms=res.inference_ms, audio_s=round(upd.audio_s, 3),
                               rtf=round(res.rtf, 4), uncertain=res.uncertain, rejected=rejected,
                               revisions=self._revisions, start_s=round(seg.start_s, 3),
                               end_s=round(seg.end_s, 3), model=res.model)
        return upd
