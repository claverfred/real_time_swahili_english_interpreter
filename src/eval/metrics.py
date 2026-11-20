"""ASR accuracy metrics. Normalization is disclosed with every result (spec section 12)."""

from __future__ import annotations

from ..asr.filters import normalize

NORMALIZATION = "NFC, lowercase, punctuation removed, whitespace collapsed"


def wer_cer(references: list[str], hypotheses: list[str]) -> dict[str, object]:
    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError('jiwer not installed (pip install -e ".[eval]")') from exc
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must align")
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    if not pairs:
        raise ValueError("all references are empty")
    r, h = zip(*pairs, strict=True)
    out = jiwer.process_words(list(r), list(h))
    return {
        "wer": round(out.wer, 4),
        "cer": round(jiwer.cer(list(r), list(h)), 4),
        "substitutions": out.substitutions, "deletions": out.deletions, "insertions": out.insertions,
        "ref_words": sum(len(x.split()) for x in r),
        "n": len(pairs),
        "empty_hypotheses": sum(1 for x in h if not x),
        "normalization": NORMALIZATION,
    }
