"""Swahili-aware sentence splitting and aggregation of ASR finals into MT units.

VAD utterances are not sentences: a max-duration split cuts mid-sentence, and one utterance
can hold several sentences. The aggregator emits complete sentences as soon as they exist,
holds a trailing fragment after a forced split (it is not a finished thought), and releases
everything at a real pause or at the end of the stream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config.schema import TextConfig

# Abbreviations after which a period does not end a sentence (lower-case, without the dot).
ABBREVIATIONS = {
    "dkt", "dk", "dak", "bw", "bi", "mh", "mhe", "prof", "nk", "n.k", "k.m", "km", "kn", "yaani",
    "sh", "tsh", "ksh", "st", "mt", "mr", "mrs", "ms", "dr", "jr", "sr", "no", "vol", "uk",
    # English (MT output is split again before TTS)
    "e.g", "i.e", "etc", "vs", "u.s", "u.k", "a.m", "p.m", "gen", "gov", "sen", "rep", "lt", "col", "inc", "ltd",
}
_BOUNDARY = re.compile(r"([.?!…]+[\"'»”’)\]]*)\s+(?=[\"'«“‘(\[]?[A-ZÀ-Ý0-9])")
_TERMINAL = re.compile(r"[.?!…][\"'»”’)\]]*$")
_CLAUSE = re.compile(r"[,;:]\s+")


def is_complete(sentence: str) -> bool:
    return bool(_TERMINAL.search(sentence.strip()))


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        before = text[start:m.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if m.group(1).startswith(".") and (last in ABBREVIATIONS or (len(last) == 1 and last.isalpha())):
            continue  # "Dkt. Mwinyi", "J. Nyerere"
        out.append(text[start:m.end(1)].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


@dataclass
class TextUnit:
    """A unit of Swahili text ready for translation."""
    unit_id: int
    text: str
    complete: bool               # ends at a sentence boundary
    uncertain: bool
    segment_ids: list[int] = field(default_factory=list)
    start_s: float = 0.0
    end_s: float = 0.0
    reason: str = "sentence"     # sentence | utterance_end | max_words | flush

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


class SentenceAggregator:
    def __init__(self, cfg: TextConfig | None = None) -> None:
        self.cfg = cfg or TextConfig()
        self._held = ""
        self._held_segments: list[int] = []
        self._held_uncertain = False
        self._held_start: float | None = None
        self._next_id = 0

    @property
    def pending(self) -> str:
        return self._held

    def _unit(self, text: str, reason: str, end_s: float) -> TextUnit:
        unit = TextUnit(self._next_id, text, is_complete(text), self._held_uncertain,
                        list(self._held_segments), self._held_start or 0.0, end_s, reason)
        self._next_id += 1
        return unit

    def _cap(self, end_s: float) -> list[TextUnit]:
        """Break a run-on fragment at the last clause boundary inside the word limit."""
        units: list[TextUnit] = []
        limit = self.cfg.max_sentence_words
        while len(self._held.split()) > limit:
            words = self._held.split()
            head = " ".join(words[:limit])
            cut = max((m.end() for m in _CLAUSE.finditer(head)), default=0)
            if cut < len(head) // 3:  # no usable clause break: cut at the word limit
                cut = len(head)
            units.append(self._unit(self._held[:cut].strip(), "max_words", end_s))
            self._held = self._held[cut:].strip()
        return units

    def push(self, text: str, *, segment_id: int, start_s: float, end_s: float,
             uncertain: bool = False, boundary: bool = True) -> list[TextUnit]:
        """Add one ASR final. ``boundary`` is False when VAD force-split the utterance."""
        text = " ".join(text.split())
        if not text:
            return self.flush(end_s) if boundary else []
        if self._held_start is None:
            self._held_start = start_s
        self._held = f"{self._held} {text}".strip()
        self._held_segments.append(segment_id)
        self._held_uncertain |= uncertain

        units: list[TextUnit] = []
        sentences = split_sentences(self._held)
        hold_tail = (not boundary and self.cfg.hold_incomplete_on_split
                     and sentences and not is_complete(sentences[-1]))
        ready = sentences[:-1] if hold_tail else sentences
        for s in ready:
            units.append(self._unit(s, "sentence" if is_complete(s) else "utterance_end", end_s))
        self._held = sentences[-1] if hold_tail else ""
        if self._held:
            if ready:  # the tail now belongs only to the latest segment
                self._held_segments = [segment_id]
                self._held_start = start_s
                self._held_uncertain = uncertain
            units += self._cap(end_s)
        else:
            self._reset_held()
        return units

    def flush(self, end_s: float = 0.0) -> list[TextUnit]:
        if not self._held:
            self._reset_held()
            return []
        unit = self._unit(self._held, "flush", end_s)
        self._reset_held()
        return [unit]

    def discard(self) -> None:
        """Drop held text without translating it (cancel). Unit ids keep increasing."""
        self._reset_held()

    def _reset_held(self) -> None:
        self._held = ""
        self._held_segments = []
        self._held_uncertain = False
        self._held_start = None
