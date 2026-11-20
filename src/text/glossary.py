"""Domain glossary: required English renderings for Swahili terms.

File formats (UTF-8):
- CSV with header ``source,target[,kind][,case_sensitive]``
- JSON or YAML list of objects with the same keys

``kind`` is ``term`` (translate as ``target``) or ``dnt`` (do not translate; target defaults to
the source). Matching is whole-word, longest entry first.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


class GlossaryError(ValueError):
    pass


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    target: str
    kind: Literal["term", "dnt"] = "term"
    case_sensitive: bool = False

    def pattern(self) -> re.Pattern[str]:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(rf"(?<![\w]){re.escape(self.source)}(?![\w])", flags)


def _truthy(v: object) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


class Glossary:
    def __init__(self, entries: list[GlossaryEntry] | None = None) -> None:
        self.entries = sorted(entries or [], key=lambda e: -len(e.source))
        self._patterns = [(e, e.pattern()) for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: str | Path) -> Glossary:
        path = Path(path)
        if not path.exists():
            raise GlossaryError(f"glossary not found: {path}")
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(text.splitlines()))
        elif path.suffix.lower() == ".json":
            rows = json.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            rows = yaml.safe_load(text) or []
        else:
            raise GlossaryError(f"unsupported glossary format: {path.suffix}")
        entries = []
        for i, row in enumerate(rows, 1):
            src = str(row.get("source", "")).strip()
            kind = str(row.get("kind") or "term").strip().lower()
            if not src:
                raise GlossaryError(f"{path}: entry {i} has no source")
            if kind not in {"term", "dnt"}:
                raise GlossaryError(f"{path}: entry {i} has invalid kind {kind!r}")
            tgt = str(row.get("target") or "").strip() or (src if kind == "dnt" else "")
            if not tgt:
                raise GlossaryError(f"{path}: entry {i} ({src}) has no target")
            entries.append(GlossaryEntry(src, tgt, kind, _truthy(row.get("case_sensitive", False))))
        return cls(entries)

    def find(self, text: str) -> list[tuple[int, int, GlossaryEntry]]:
        """Non-overlapping matches, longest entries winning."""
        taken: list[tuple[int, int, GlossaryEntry]] = []
        for entry, pat in self._patterns:
            for m in pat.finditer(text):
                if all(m.end() <= s or m.start() >= e for s, e, _ in taken):
                    taken.append((m.start(), m.end(), entry))
        return sorted(taken, key=lambda t: t[0])

    def missing_targets(self, source: str, target: str) -> list[str]:
        """Glossary targets required by ``source`` that do not appear in ``target``."""
        low = target.lower()
        out = []
        for _, _, e in self.find(source):
            present = e.target in target if e.case_sensitive else e.target.lower() in low
            if not present:
                out.append(e.target)
        return out
