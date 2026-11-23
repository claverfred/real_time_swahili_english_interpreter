"""ASR updates -> stabilized Swahili -> sentence units -> English. Synchronous core of the
translation queue; Phase 6 runs it as an async task.

Policy:
- ASR finals are stabilized, aggregated into sentence units and translated with beam search.
  Context is the last ``mt.context_sentences`` (Swahili, English) pairs.
- With ``mt.translate_partials``, the committed (stable) ASR prefix is translated greedily
  as a *provisional* result once it grows by ``mt.partial_min_new_words`` words. The unstable
  tail is never translated, and provisional output is never fed back as context.
- Uncertain transcripts are translated and flagged, or skipped (``mt.uncertain_policy``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..asr.streaming import AsrUpdate
from ..config.schema import MtConfig, TextConfig
from ..pipeline.events import EventRecorder, EventType
from ..text.sentences import SentenceAggregator, TextUnit
from ..text.stabilize import stabilize
from .translator import Translator
from .types import MtResult


@dataclass
class MtUpdate:
    kind: Literal["partial", "final", "skipped"]
    source: str
    result: MtResult | None
    unit: TextUnit | None = None
    segment_id: int | None = None
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.result.text if self.result else ""


class MtStage:
    def __init__(self, translator: Translator, mt_cfg: MtConfig, text_cfg: TextConfig | None = None,
                 recorder: EventRecorder | None = None) -> None:
        self.translator, self.cfg = translator, mt_cfg
        self.text_cfg = text_cfg or TextConfig()
        self.recorder = recorder
        self.aggregator = SentenceAggregator(self.text_cfg)
        self.history: list[tuple[str, str]] = []
        self._partial_words = 0
        self._partial_segment: int | None = None

    def _emit(self, etype: EventType, **kw) -> None:
        if self.recorder is not None:
            self.recorder.emit(etype, **kw)

    def _translate_unit(self, unit: TextUnit) -> MtUpdate:
        if unit.uncertain and self.cfg.uncertain_policy == "skip":
            self._emit(EventType.STATUS, segment_id=unit.segment_ids[-1] if unit.segment_ids else None,
                       text=unit.text, status="mt_skipped_uncertain", unit_id=unit.unit_id)
            return MtUpdate("skipped", unit.text, None, unit, extra={"reason": "uncertain"})
        seg = unit.segment_ids[-1] if unit.segment_ids else None
        self._emit(EventType.MT_REQUESTED, segment_id=seg, text=unit.text, unit_id=unit.unit_id,
                   kind="final", complete=unit.complete)
        res = self.translator.translate(unit.text, context=self.history, uncertain=unit.uncertain)
        self._emit(EventType.MT_FINAL, segment_id=seg, text=res.text, source=unit.text, unit_id=unit.unit_id,
                   latency_ms=res.latency_ms, cached=res.cached, flags=res.flags,
                   context_used=res.context_used, model=res.model, start_s=unit.start_s, end_s=unit.end_s)
        if res.text:
            self.history = (self.history + [(unit.text, res.text)])[-max(self.cfg.context_sentences, 1):]
        return MtUpdate("final", unit.text, res, unit, seg)

    def on_asr_update(self, up: AsrUpdate) -> list[MtUpdate]:
        if up.kind == "partial":
            return self._on_partial(up)
        self._partial_words, self._partial_segment = 0, None
        boundary = up.extra.get("vad_reason") != "max_duration"
        st = stabilize(up.text, self.text_cfg, final=boundary, continuation=bool(self.aggregator.pending))
        uncertain = bool(up.result.uncertain) if up.result is not None else bool(up.extra.get("uncertain"))
        if st.text:
            self._emit(EventType.TRANSCRIPT_STABILIZED, segment_id=up.segment_id, text=st.text,
                       original=st.original, changes=st.changes, uncertain=uncertain)
        units = self.aggregator.push(st.text, segment_id=up.segment_id, start_s=up.start_s, end_s=up.end_s,
                                     uncertain=uncertain, boundary=boundary)
        return [self._translate_unit(u) for u in units]

    def _on_partial(self, up: AsrUpdate) -> list[MtUpdate]:
        if not self.cfg.translate_partials or not up.stable:
            return []
        if up.segment_id != self._partial_segment:
            self._partial_segment, self._partial_words = up.segment_id, 0
        source = f"{self.aggregator.pending} {up.stable}".strip()
        n = len(source.split())
        if n - self._partial_words < self.cfg.partial_min_new_words:
            return []
        self._partial_words = n
        self._emit(EventType.MT_REQUESTED, segment_id=up.segment_id, text=source, kind="partial")
        res = self.translator.translate(source, provisional=True)
        self._emit(EventType.MT_PARTIAL, segment_id=up.segment_id, text=res.text, source=source,
                   latency_ms=res.latency_ms, flags=res.flags, provisional=True)
        return [MtUpdate("partial", source, res, None, up.segment_id)]

    def flush(self, end_s: float = 0.0) -> list[MtUpdate]:
        return [self._translate_unit(u) for u in self.aggregator.flush(end_s)]

    def cancel(self) -> None:
        """Discard held, untranslated text; keep translation context."""
        self.aggregator.discard()
        self._partial_words, self._partial_segment = 0, None

    def reset(self) -> None:
        """Cancel and forget translation context (new conversation). Unit ids stay unique."""
        self.cancel()
        self.history = []
