"""Numbers, acronyms and names: extraction, preservation checks and placeholder masking.

Checks report what is *missing* from a translation. Some losses are legitimate (Swahili
"UKIMWI" is English "AIDS", "Marekani" is "America"), so they are flags for evaluation and
review, not automatic corrections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.,])[-+]?\d+(?:[.,:/]\d+)*%?(?![\w])")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_SENT_START = re.compile(r"(?:^|[.?!…][\"'»”’)\]]*\s+)[\"'«“‘(\[]?([^\W\d_][\w'’-]*)")


def canonical_number(token: str) -> str:
    """'2,000' == '2000' == '2 000'; '2,5' (decimal comma) == '2.5'; keeps % and sign."""
    t = token.strip()
    pct = t.endswith("%")
    t = t.rstrip("%")
    if re.fullmatch(r"[-+]?\d{1,3}(?:[,.\s]\d{3})+", t):
        t = re.sub(r"[,.\s]", "", t)
    elif re.fullmatch(r"[-+]?\d+,\d{1,2}", t):
        t = t.replace(",", ".")
    return t + ("%" if pct else "")


def numbers(text: str) -> list[str]:
    return [canonical_number(m.group()) for m in _NUMBER.finditer(text)]


def acronyms(text: str) -> list[str]:
    return [m.group() for m in _ACRONYM.finditer(text)]


def names(text: str) -> list[str]:
    """Capitalized words that do not start a sentence: a cheap proper-noun proxy."""
    starts = {m.start(1) for m in _SENT_START.finditer(text)}
    out = []
    for m in _WORD.finditer(text):
        w = m.group()
        if m.start() in starts or not w[0].isupper() or _ACRONYM.fullmatch(w):
            continue
        out.append(w)
    return out


def _missing(src_items: list[str], tgt_items: list[str]) -> list[str]:
    pool = list(tgt_items)
    missing = []
    for item in src_items:
        if item in pool:
            pool.remove(item)
        else:
            missing.append(item)
    return missing


def check_preservation(source: str, target: str) -> dict[str, dict[str, list[str]]]:
    tgt_low = target.lower()
    src_names = names(source)
    return {
        "numbers": {"source": numbers(source), "missing": _missing(numbers(source), numbers(target))},
        "acronyms": {"source": acronyms(source),
                     "missing": [a for a in acronyms(source) if a not in target]},
        "names": {"source": src_names, "missing": [n for n in src_names if n.lower() not in tgt_low]},
    }


# Placeholder masking -------------------------------------------------------------------------

@dataclass
class Masked:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> replacement


def placeholder(i: int) -> str:
    # Chosen empirically (scripts/mt_placeholder_probe.py, 2026-11-20): "Z1Q" survived 6/6 sentences on
    # both NLLB-600M and opus-mt-swc-en; "X1", "NUM1", "<1>", "#1", "__1__" were each dropped at least once.
    return f"Z{i + 1}Q"


def mask(text: str, spans: list[tuple[int, int, str]]) -> Masked:
    """Replace non-overlapping (start, end, replacement) spans with placeholders."""
    out, mapping, pos = [], {}, 0
    for start, end, repl in sorted(spans):
        if start < pos:
            continue
        ph = placeholder(len(mapping))
        out += [text[pos:start], ph]
        mapping[ph] = repl
        pos = end
    out.append(text[pos:])
    return Masked("".join(out), mapping)


def unmask(text: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    """Restore placeholders. Returns (text, placeholders the model dropped)."""
    missing = []
    for ph, repl in sorted(mapping.items(), key=lambda kv: -len(kv[0])):  # X10 before X1
        pattern = re.compile(rf"(?<![\w]){re.escape(ph)}(?![\w])")
        if not pattern.search(text):
            missing.append(ph)
            continue
        text = pattern.sub(lambda _m, r=repl: r, text, count=1)
    return text, missing


def number_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _NUMBER.finditer(text)]
