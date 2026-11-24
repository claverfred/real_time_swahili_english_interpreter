"""Transcript stabilization: surface clean-up of ASR text before translation.

Only punctuation, spacing, case and runaway repeats are changed. Words are never added,
guessed or completed. Every change is listed so evaluation can audit it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..asr.filters import collapse_repeats
from ..config.schema import TextConfig
from .sentences import ABBREVIATIONS

_INNER_START = re.compile(r"([.?!…][\"'»”’)\]]*\s+[\"'«“‘(\[]?)([^\W\d_])")

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!…])")
_MISSING_SPACE_AFTER = re.compile(r"([,;:?!])(?=[^\W\d_])")
_REPEAT_PUNCT = re.compile(r"([,;:?!])\1+|(?<!\.)\.\.(?!\.)")
_EDGE_ELLIPSIS = re.compile(r"^(?:\.{2,}|…)\s*|\s*(?:\.{3,}|…)$")
_TERMINAL = re.compile(r"[.?!…][\"'»”’)\]]*$")
# Swahili interrogatives that reliably mark a question when they open the utterance.
_QUESTION_OPENERS = {"je", "nani", "nini", "wapi", "lini", "vipi", "kwa nini", "mbona", "kwanini"}


@dataclass
class StabilizedText:
    text: str
    original: str
    changes: list[str] = field(default_factory=list)


def _is_question(text: str) -> bool:
    low = text.lower()
    return any(low == q or low.startswith(q + " ") or low.startswith(q + ",") for q in _QUESTION_OPENERS)


def _sentence_case(text: str) -> str:
    def fix(m: re.Match[str]) -> str:
        before = text[:m.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if m.group(1).startswith(".") and (last in ABBREVIATIONS or len(last) == 1):
            return m.group(0)
        return m.group(1) + m.group(2).upper()
    return _INNER_START.sub(fix, text)


def stabilize(text: str, cfg: TextConfig | None = None, *, final: bool = True,
              continuation: bool = False) -> StabilizedText:
    """``final``: the utterance ended at a real pause. ``continuation``: the text continues a
    sentence held from the previous utterance, so its first letter keeps its case."""
    cfg = cfg or TextConfig()
    original = text
    changes: list[str] = []
    out = unicodedata.normalize("NFC", text)
    out = " ".join(out.split())

    trimmed = _EDGE_ELLIPSIS.sub("", out).strip()
    if trimmed != out:
        changes.append("edge_ellipsis_removed")
        out = trimmed

    collapsed = collapse_repeats(out)
    if collapsed != out:
        changes.append("repeat_collapsed")
        out = collapsed

    if cfg.number_words_to_digits:
        from .sw_numbers import number_words_to_digits

        digits, n = number_words_to_digits(out)
        if n:
            changes.append("number_words_to_digits")
            out = digits

    spaced = _MISSING_SPACE_AFTER.sub(r"\1 ", _SPACE_BEFORE_PUNCT.sub(r"\1", out))
    spaced = _REPEAT_PUNCT.sub(lambda m: m.group(1) or ".", spaced)
    if spaced != out:
        changes.append("punctuation_spacing")
        out = spaced

    if not out:
        return StabilizedText("", original, changes)

    if cfg.capitalize:
        cased = _sentence_case(out)
        if not continuation and cased[0].islower():
            cased = cased[0].upper() + cased[1:]
        if cased != out:
            changes.append("capitalized")
            out = cased

    if final and cfg.add_final_punct and not _TERMINAL.search(out):
        out = out.rstrip(",;:") + ("?" if _is_question(out) else ".")
        changes.append("final_punct_added")

    return StabilizedText(out, original, changes)
