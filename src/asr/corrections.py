"""Swahili -> Swahili fixes for recurring ASR mistakes, applied before translation.

The list holds words or phrases the recognizer keeps getting wrong (``mwenye kiti`` for
``mwenyekiti``, ``madam barimbali`` for ``mambo mbalimbali``). It is a stop-gap for systematic
errors, not a language model: add an entry only after seeing the same mistake more than once,
and prefer multi-word entries over single words so a correct use elsewhere is not rewritten.

File format: UTF-8 CSV with header ``heard,correct[,note]``. Blank lines and lines starting with
``#`` are ignored. Matching is whole-word and case-insensitive, longest entry first; a match at the
start of a capitalised sentence keeps the capital.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


class CorrectionsError(ValueError):
    pass


@dataclass(frozen=True)
class Correction:
    heard: str
    correct: str


class AsrCorrections:
    def __init__(self, entries: list[Correction] | None = None) -> None:
        self.entries = sorted(entries or [], key=lambda e: -len(e.heard))
        if self.entries:
            alt = "|".join(re.escape(e.heard) for e in self.entries)
            self._pattern: re.Pattern[str] | None = re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)
        else:
            self._pattern = None
        self._lookup = {e.heard.lower(): e.correct for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: str | Path) -> AsrCorrections:
        path = Path(path)
        if not path.exists():
            raise CorrectionsError(f"ASR corrections file not found: {path}")
        lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        entries, seen = [], set()
        for i, row in enumerate(csv.DictReader(lines), 2):
            heard = " ".join(str(row.get("heard") or "").split())
            correct = " ".join(str(row.get("correct") or "").split())
            if not heard or not correct:
                raise CorrectionsError(f"{path}: row {i} needs both 'heard' and 'correct'")
            if heard.lower() in seen:
                raise CorrectionsError(f"{path}: row {i} repeats {heard!r}")
            seen.add(heard.lower())
            entries.append(Correction(heard, correct))
        return cls(entries)

    def apply(self, text: str) -> tuple[str, int]:
        """Return the corrected text and how many replacements were made (one pass, no chaining)."""
        if not self._pattern or not text:
            return text, 0
        count = 0

        def sub(m: re.Match[str]) -> str:
            nonlocal count
            count += 1
            fix = self._lookup[m.group(0).lower()]
            # keep the sentence's capital letter
            return fix[:1].upper() + fix[1:] if m.group(0)[:1].isupper() else fix

        return self._pattern.sub(sub, text), count
