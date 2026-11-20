"""Hallucination suppression, repetition removal, and overlap merging for ASR output.

The policy is conservative: text is *removed or flagged*, never invented or completed.
Every removal records a reason on the segment so it can be audited and evaluated.
"""

from __future__ import annotations

import re
import unicodedata

from ..config.schema import AsrConfig
from .types import AsrSegment

# Whisper's well-known training-data hallucinations (subtitle credits, YouTube outros).
# Matched on normalized text. Swahili variants observed on silence/music are included.
HALLUCINATION_PHRASES = [
    "thanks for watching", "thank you for watching", "please subscribe", "subtitles by",
    "amara org", "transcribed by", "translated by", "like and subscribe",
    "asante kwa kutazama", "asanteni kwa kutazama", "usisahau ku subscribe", "tafadhali subscribe",
    "manukuu na", "imetafsiriwa na",
]

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, NFC, strip punctuation — for comparisons only, never for output."""
    t = unicodedata.normalize("NFC", text).lower()
    t = _PUNCT.sub(" ", t).replace("'", " ")
    return _SPACE.sub(" ", t).strip()


def words_of(text: str) -> list[str]:
    return text.split()


def collapse_repeats(text: str, max_repeats: int = 2, max_ngram: int = 6) -> str:
    """Collapse runaway n-gram loops ("na na na na na" -> "na na"). Keeps original casing."""
    toks = text.split()
    for n in range(1, max_ngram + 1):
        norm = [normalize(t) for t in toks]
        out: list[str] = []
        i = 0
        while i < len(toks):
            gram = norm[i:i + n]
            reps = 1
            if len(gram) == n:
                while norm[i + reps * n:i + (reps + 1) * n] == gram:
                    reps += 1
            if reps > max_repeats:
                out.extend(toks[i:i + max_repeats * n])
                i += reps * n
            else:
                out.append(toks[i])
                i += 1
        toks = out
    return " ".join(toks)


def is_hallucination_phrase(text: str) -> bool:
    norm = normalize(text)
    return bool(norm) and any(p in norm for p in HALLUCINATION_PHRASES)


def filter_segments(segments: list[AsrSegment], cfg: AsrConfig, previous_text: str = "") -> list[AsrSegment]:
    """Mark segments rejected/uncertain in place and return them.

    Rules, in order:
      1. empty text                                     -> rejected "empty"
      2. no_speech_prob > threshold and low logprob      -> rejected "no_speech"
      3. compression ratio > threshold (repetition loop) -> rejected "repetition_loop"
      4. known subtitle/outro hallucination phrase        -> rejected "hallucination_phrase"
      5. identical to the previous kept segment          -> rejected "duplicate_segment"
      6. avg_logprob < log_prob_threshold - 0.5          -> rejected "very_low_confidence"
      7. avg_logprob < log_prob_threshold                -> kept, flagged uncertain
    Kept text has runaway n-gram repeats collapsed.
    """
    prev_norm = normalize(previous_text)
    for seg in segments:
        seg.text = seg.text.strip()
        norm = normalize(seg.text)
        if not norm:
            seg.rejected = "empty"
        elif seg.no_speech_prob > cfg.no_speech_threshold and seg.avg_logprob < cfg.log_prob_threshold:
            seg.rejected = "no_speech"
        elif seg.compression_ratio > cfg.compression_ratio_threshold:
            seg.rejected = "repetition_loop"
        elif is_hallucination_phrase(seg.text):
            seg.rejected = "hallucination_phrase"
        elif prev_norm and norm == prev_norm:
            seg.rejected = "duplicate_segment"
        elif seg.avg_logprob < cfg.log_prob_threshold - 0.5:
            seg.rejected = "very_low_confidence"
        else:
            seg.uncertain = seg.avg_logprob < cfg.log_prob_threshold
            seg.text = collapse_repeats(seg.text)
            prev_norm = normalize(seg.text)
    return segments


def merge_overlap(previous: str, new: str, max_words: int = 12) -> str:
    """Drop the prefix of ``new`` that repeats the suffix of ``previous``.

    Used when consecutive segments share ``asr.overlap_s`` of audio (after a max-duration split).
    """
    prev = previous.split()
    cur = new.split()
    if not prev or not cur:
        return new
    pn = [normalize(w) for w in prev[-max_words:]]
    cn = [normalize(w) for w in cur[:max_words]]
    for k in range(min(len(pn), len(cn)), 0, -1):
        if pn[-k:] == cn[:k]:
            return " ".join(cur[k:])
    return new


def common_prefix(a: list[str], b: list[str]) -> int:
    """Length of the common word prefix, comparing normalized forms."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if normalize(x) != normalize(y):
            break
        n += 1
    return n
