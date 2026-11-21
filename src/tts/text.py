"""English text preparation for TTS: spoken-form normalization and sentence-safe chunking.

Chunks never split inside a sentence unless the sentence exceeds ``chunk_max_chars``; then the
split is at a clause boundary (``, ; : —``), and only as a last resort at a word boundary.
The first chunk of an utterance may be cut at an early clause boundary so audio starts sooner.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import yaml

from ..text.sentences import split_sentences

# Acronyms normally read as words, not letters. Extend through the pronunciation dictionary.
WORD_ACRONYMS = {"NATO", "UNESCO", "UNICEF", "AIDS", "COVID", "NASA", "FIFA", "OPEC", "SADC", "UKIMWI",
                 "TANESCO", "CAF", "UEFA", "ASEAN", "OK"}
_ACRONYM = re.compile(r"\b[A-Z]{2,5}\b")
_CLAUSE = re.compile(r"(?<=[,;:—–])\s+|\s+(?=[—–]\s)")
_CURRENCY = re.compile(r"\b(TSh|Tsh|TZS|KSh|Ksh|KES|USh|UGX)\.?\s?(\d(?:[\d,.]*\d)?)")
_CURRENCY_NAMES = {"tsh": "Tanzanian shillings", "tzs": "Tanzanian shillings", "ksh": "Kenyan shillings",
                   "kes": "Kenyan shillings", "ush": "Ugandan shillings", "ugx": "Ugandan shillings"}


class PronunciationError(ValueError):
    pass


class PronunciationDict:
    """word -> spoken form, matched whole-word and case-insensitively (longest first)."""

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self.entries = dict(entries or {})
        words = sorted(self.entries, key=len, reverse=True)
        self._lookup = {w.lower(): s for w, s in self.entries.items()}
        self._re = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, words)) + r")(?![\w])",
                              re.IGNORECASE) if words else None

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._lookup

    @classmethod
    def load(cls, path: str | Path) -> PronunciationDict:
        path = Path(path)
        if not path.exists():
            raise PronunciationError(f"pronunciation dictionary not found: {path}")
        text = path.read_text(encoding="utf-8-sig")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows = list(csv.DictReader(text.splitlines()))
            data = {r["word"].strip(): r["spoken"].strip() for r in rows if r.get("word")}
        elif suffix == ".json":
            data = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        else:
            raise PronunciationError(f"unsupported pronunciation dictionary format: {suffix}")
        if not isinstance(data, dict) or not all(isinstance(v, str) and v for v in data.values()):
            raise PronunciationError(f"{path}: expected a mapping of word -> spoken form")
        return cls(data)

    def apply(self, text: str) -> str:
        if self._re is None:
            return text
        return self._re.sub(lambda m: self._lookup[m.group(1).lower()], text)


def normalize_for_tts(text: str, *, pron: PronunciationDict | None = None, spell_acronyms: bool = True) -> str:
    """Spoken form: dictionary entries, currency, symbols and letter-by-letter acronyms."""
    out = " ".join(text.split())
    out = _CURRENCY.sub(lambda m: f"{m.group(2)} {_CURRENCY_NAMES[m.group(1).lower().rstrip('.')]}", out)
    out = re.sub(r"(\d)\s?%", r"\1 percent", out)
    out = out.replace("&", " and ")
    if spell_acronyms:
        def spell(m: re.Match[str]) -> str:
            w = m.group()
            if w in WORD_ACRONYMS or (pron is not None and w in pron):
                return w
            return " ".join(w)
        out = _ACRONYM.sub(spell, out)
    if pron is not None:  # last, so respellings like "Doh-DOH-mah" are not altered
        out = pron.apply(out)
    return " ".join(out.split())


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Split one over-long sentence at clause boundaries, then at word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    cur = ""
    for clause in _CLAUSE.split(sentence):
        if not clause:
            continue
        if cur and len(cur) + 1 + len(clause) > max_chars:
            parts.append(cur)
            cur = clause
        else:
            cur = f"{cur} {clause}".strip()
    if cur:
        parts.append(cur)
    out: list[str] = []
    for p in parts:
        while len(p) > max_chars:  # no clause boundary small enough: break at a word
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def chunk_text(text: str, max_chars: int = 220, first_max_chars: int | None = 80) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    for s in sentences:
        chunks += _split_long(s, max_chars)
    if chunks and first_max_chars and len(chunks[0]) > first_max_chars:
        first = chunks[0]
        cut = max((m.start() for m in _CLAUSE.finditer(first) if 20 <= m.start() <= first_max_chars), default=-1)
        if cut > 0:  # only at a clause boundary: an early break must not hurt prosody
            chunks[0:1] = [first[:cut].strip(), first[cut:].strip()]
    return [c for c in chunks if c]
