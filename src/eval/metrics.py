"""Evaluation metrics. Normalization and metric methods are disclosed with every result (spec 12-14).

ASR: WER/CER on normalized text, number and name recall, empty and rejected-output rates.
MT and speech-to-English: BLEU, chrF, chrF++ (sacrebleu signatures recorded), preservation of
numbers / names / acronyms against the *human reference*, and length-ratio proxies for
omissions and hallucinated additions. Bootstrap confidence intervals resample utterances.
References are always human; a hypothesis is never used as a reference.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..asr.filters import normalize
from ..text.entities import acronyms, names, numbers
from ..text.sw_numbers import sw_numbers_to_digits  # noqa: F401  (re-exported)

NORMALIZATION = "NFC, lowercase, punctuation removed, whitespace collapsed"
# Length-ratio proxies (hypothesis words / reference words, per sentence). Disclosed, not tuned.
OMISSION_RATIO = 0.6
ADDITION_RATIO = 1.6


NUMBER_NORMALIZATION = ("Swahili cardinal number words converted to digits in reference and hypothesis "
                        "(phrases containing a tens or scale word, e.g. 'elfu moja mia tisa sitini' -> 1960)")


def _jiwer():
    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError('jiwer not installed (pip install -e ".[eval]")') from exc
    return jiwer


def wer_cer(references: list[str], hypotheses: list[str]) -> dict[str, object]:
    jiwer = _jiwer()
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


def utterance_errors(reference: str, hypothesis: str) -> dict[str, int]:
    """Word and character edit counts for one utterance (for bootstrap and failure lists)."""
    jiwer = _jiwer()
    r, h = normalize(reference), normalize(hypothesis)
    if not r:
        return {"word_errors": len(h.split()), "ref_words": 0, "char_errors": len(h), "ref_chars": 0}
    w = jiwer.process_words(r, h)
    c = jiwer.process_characters(r, h)
    return {"word_errors": w.substitutions + w.deletions + w.insertions, "ref_words": len(r.split()),
            "char_errors": c.substitutions + c.deletions + c.insertions, "ref_chars": len(r)}


def bootstrap_ratio(errors: Sequence[int], totals: Sequence[int], *, n_boot: int = 1000, seed: int = 0,
                    ci: float = 95.0) -> dict[str, float]:
    """Percentile bootstrap CI of sum(errors)/sum(totals), resampling utterances."""
    if not errors or sum(totals) == 0:
        return {"low": float("nan"), "high": float("nan"), "n_boot": 0}
    rng = random.Random(seed)
    n = len(errors)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        t = sum(totals[i] for i in idx)
        stats.append(sum(errors[i] for i in idx) / t if t else 0.0)
    stats.sort()
    lo = stats[int((100 - ci) / 200 * (n_boot - 1))]
    hi = stats[int((1 - (100 - ci) / 200) * (n_boot - 1))]
    return {"low": round(lo, 4), "high": round(hi, 4), "n_boot": n_boot, "ci": ci}


def recall(references: Sequence[str], hypotheses: Sequence[str], kind: str) -> dict[str, object]:
    """Share of reference numbers / names / acronyms that appear in the hypothesis.

    Numbers are compared in canonical form (2,000 == 2000); names and acronyms
    case-insensitively as substrings, so inflected forms still count.
    """
    extract = {"numbers": numbers, "names": names, "acronyms": acronyms}[kind]
    found = total = 0
    missed: list[str] = []
    for ref, hyp in zip(references, hypotheses, strict=True):
        items = extract(ref)
        if not items:
            continue
        if kind == "numbers":
            pool = list(numbers(hyp))
            for x in items:
                total += 1
                if x in pool:
                    pool.remove(x)
                    found += 1
                else:
                    missed.append(x)
        else:
            low = hyp.lower()
            for x in items:
                total += 1
                if x.lower() in low:
                    found += 1
                else:
                    missed.append(x)
    return {"recall": round(found / total, 4) if total else None, "items": total, "found": found,
            "examples_missed": missed[:15]}


def length_ratio_rates(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, object]:
    """Per-sentence length ratio: proxies for omissions (too short) and additions (too long)."""
    ratios = []
    for r, h in zip(references, hypotheses, strict=True):
        rw = len(r.split())
        if rw:
            ratios.append(len(h.split()) / rw)
    n = len(ratios)
    return {"omission_rate": round(sum(x < OMISSION_RATIO for x in ratios) / n, 4) if n else None,
            "addition_rate": round(sum(x > ADDITION_RATIO for x in ratios) / n, 4) if n else None,
            "empty_rate": round(sum(x == 0 for x in ratios) / n, 4) if n else None,
            "method": f"hyp/ref word ratio < {OMISSION_RATIO} = omission, > {ADDITION_RATIO} = addition "
                      "(proxy; human review decides)"}


def mt_scores(hypotheses: list[str], references: list[str], *, bootstrap: int = 0, seed: int = 0) -> dict[str, object]:
    """Corpus BLEU / chrF / chrF++ with sacrebleu signatures and optional bootstrap CIs."""
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError('sacrebleu not installed (pip install -e ".[eval]")') from exc
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must align")
    bleu, chrf, chrfpp = sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF(), sacrebleu.metrics.CHRF(word_order=2)
    out: dict[str, object] = {
        "bleu": round(bleu.corpus_score(hypotheses, [references]).score, 2),
        "chrf": round(chrf.corpus_score(hypotheses, [references]).score, 2),
        "chrf_pp": round(chrfpp.corpus_score(hypotheses, [references]).score, 2),
        "n": len(hypotheses),
        "signatures": {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature()),
                       "chrf_pp": str(chrfpp.get_signature())},
    }
    if bootstrap and len(hypotheses) > 1:
        rng = random.Random(seed)
        n = len(hypotheses)
        b_s, c_s = [], []
        for _ in range(bootstrap):
            idx = [rng.randrange(n) for _ in range(n)]
            h, r = [hypotheses[i] for i in idx], [references[i] for i in idx]
            b_s.append(bleu.corpus_score(h, [r]).score)
            c_s.append(chrf.corpus_score(h, [r]).score)
        b_s.sort()
        c_s.sort()
        lo, hi = int(0.025 * (bootstrap - 1)), int(0.975 * (bootstrap - 1))
        out["ci95"] = {"bleu": [round(b_s[lo], 2), round(b_s[hi], 2)],
                       "chrf": [round(c_s[lo], 2), round(c_s[hi], 2)], "n_boot": bootstrap}
    return out


def sentence_chrf(hypothesis: str, reference: str) -> float:
    import sacrebleu

    return round(sacrebleu.sentence_chrf(hypothesis, [reference]).score, 2)
