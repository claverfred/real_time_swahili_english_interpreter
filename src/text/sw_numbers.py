"""Swahili cardinal number words -> digits ("elfu moja mia tisa sitini" -> "1960").

Used by evaluation (so number formats don't count as recognition errors) and, optionally, before
translation (``text.number_words_to_digits``) so the English keeps exact figures.
Only phrases containing a tens or scale word are converted: lone unit words ("tatu", "moja") are
often ordinals ("karne ya tatu" = third century) or adjectives, so they are left alone.
"""

from __future__ import annotations

import re

UNITS = {"sifuri": 0, "moja": 1, "mbili": 2, "tatu": 3, "nne": 4, "tano": 5, "sita": 6, "saba": 7,
         "nane": 8, "tisa": 9}
TENS = {"kumi": 10, "ishirini": 20, "thelathini": 30, "arobaini": 40, "hamsini": 50, "sitini": 60,
        "sabini": 70, "themanini": 80, "tisini": 90}
SCALES = {"mia": 100, "elfu": 1000, "laki": 100_000, "milioni": 1_000_000, "bilioni": 1_000_000_000}
_VOCAB = set(UNITS) | set(TENS) | set(SCALES) | {"na"}
_TOKEN = re.compile(r"\S+")
_EDGE = re.compile(r"^([\"'(\[«“‘]*)(.*?)([\"')\]»”’.,;:!?]*)$")


def parse(words: list[str]) -> int | None:
    """Value of a Swahili cardinal phrase; ``None`` if it contains a non-number word.

    The multiplier follows the scale word: one unit for "mia" (mia tisa = 900); units/tens without
    "na" for larger scales (elfu ishirini = 20000). "na" or another scale word starts the next
    additive term: "elfu moja mia tisa sitini" = 1000 + 900 + 60.
    """
    total, i, n = 0, 0, len(words)
    while i < n:
        w = words[i]
        if w == "na":
            i += 1
        elif w in SCALES:
            scale, j, mult, got = SCALES[w], i + 1, 0, False
            if scale == 100:
                if j < n and words[j] in UNITS:
                    mult, j, got = UNITS[words[j]], j + 1, True
            else:
                # "elfu mia tatu thelathini" = 330 x 1000: a hundreds multiplier right after the scale
                if j + 1 < n and words[j] == "mia" and words[j + 1] in UNITS:
                    mult, j, got = 100 * UNITS[words[j + 1]], j + 2, True
                while j < n and (words[j] in UNITS or words[j] in TENS):
                    mult += UNITS.get(words[j], 0) + TENS.get(words[j], 0)
                    j, got = j + 1, True
            total += scale * (mult if got else 1)
            i = j
        elif w in TENS:
            total += TENS[w]
            i += 1
        elif w in UNITS:
            total += UNITS[w]
            i += 1
        else:
            return None
    return total


def number_words_to_digits(text: str) -> tuple[str, int]:
    """Replace number-word phrases by digits, keeping surrounding punctuation and other text.
    Returns (text, number of phrases converted). A phrase never spans sentence punctuation."""
    toks = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text)]
    parts = [_EDGE.match(t[2]).groups() for t in toks]           # (lead, core, trail)
    out, last, i, converted = [], 0, 0, 0
    while i < len(toks):
        core = parts[i][1].lower()
        decimal_start = (core in UNITS and not parts[i][2] and i + 2 < len(toks)
                         and parts[i + 1][1].lower() == "nukta" and parts[i + 2][1].lower() in UNITS)
        if core in TENS or core in SCALES or decimal_start:
            j = i
            while j < len(toks) and parts[j][1].lower() in _VOCAB | {"nukta"}:
                if j > i and parts[j][0]:
                    break                                             # opening quote/bracket: stop
                j += 1
                if parts[j - 1][2]:                                   # punctuation ends the phrase
                    break
            while j > i and parts[j - 1][1].lower() == "na":          # trailing "na" is a conjunction
                j -= 1
            words = [p[1].lower() for p in parts[i:j]]
            value: int | str | None = None
            if "nukta" in words:                                      # decimal: digits after "nukta"
                k = words.index("nukta")
                frac = words[k + 1:]
                if k == 0 or not frac or any(w not in UNITS for w in frac):
                    j, words = i + k, words[:k]                       # keep only the integer part
                    if not any(w in TENS or w in SCALES for w in words):
                        words = []
                else:
                    whole = parse(words[:k])
                    value = None if whole is None else f"{whole}." + "".join(str(UNITS[w]) for w in frac)
                    words = []
            if words:
                value = parse(words)
            if value is not None:
                out.append(text[last:toks[i][0]])
                out.append(parts[i][0] + str(value) + parts[j - 1][2])
                last = toks[j - 1][1]
                converted += 1
                i = j
                continue
        i += 1
    out.append(text[last:])
    return "".join(out), converted


def sw_numbers_to_digits(text: str) -> str:
    """Convenience: converted text only."""
    return number_words_to_digits(text)[0]
