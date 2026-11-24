"""Human evaluation kit (spec 13-14): rating sheets for bilingual raters and a scorer.

Automatic metrics compare against one human reference and cannot judge meaning or naturalness.
The sheet asks Swahili-English bilingual raters to judge the system's English against what was
*said* (the Swahili), on the scales in docs/human_evaluation.md. Items are shuffled with a fixed
seed; system internals (ASR text, metric scores) are hidden from raters. The English reference is
kept in a separate key file so it cannot bias the rating.
"""

from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RATING_COLUMNS = ["adequacy_1_5", "fluency_1_5", "names_ok_y_n_na", "numbers_ok_y_n_na", "terminology_ok_y_n_na",
                  "error_severity_none_minor_major_critical", "naturalness_1_5", "comment"]
SHEET_COLUMNS = ["item", "swahili_said", "english_output", "audio_file", *RATING_COLUMNS]
SEVERITY = ["none", "minor", "major", "critical"]


def make_sheet(rows: list[dict[str, Any]], out_dir: str | Path, *, n: int = 60, seed: int = 8,
               synthesize: Any = None, sample_rate: int = 22050) -> dict[str, Any]:
    """Write ``rating_sheet.csv`` (for raters) and ``key.csv`` (for the analyst).

    ``rows`` are per-item results of a speech evaluation (``sw_ref``, ``en_hyp``, ``en_ref``).
    ``synthesize(text) -> np.ndarray`` adds an English audio file per item for naturalness ratings.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = [r for r in rows if r.get("en_hyp")]
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(n, len(pool)))
    sheet, key = [], []
    for i, r in enumerate(chosen, 1):
        item = f"H{i:03d}"
        audio_name = ""
        if synthesize is not None:
            from ..audio import save_wav

            audio_name = f"audio/{item}.wav"
            save_wav(out / audio_name, synthesize(r["en_hyp"]), sample_rate)
        sheet.append({"item": item, "swahili_said": r["sw_ref"], "english_output": r["en_hyp"],
                      "audio_file": audio_name, **{c: "" for c in RATING_COLUMNS}})
        key.append({"item": item, "source_item_id": r.get("item_id"), "condition": r.get("condition"),
                    "english_reference": r.get("en_ref"), "asr_text": r.get("sw_hyp"), "asr_wer": r.get("wer"),
                    "sentence_chrf": r.get("chrf")})
    _write_csv(out / "rating_sheet.csv", SHEET_COLUMNS, sheet)
    _write_csv(out / "key.csv", list(key[0]) if key else ["item"], key)
    return {"items": len(sheet), "sheet": str(out / "rating_sheet.csv"), "key": str(out / "key.csv"),
            "audio": synthesize is not None, "seed": seed}


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:      # BOM: opens cleanly in Excel
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def _num(v: str) -> float | None:
    try:
        x = float(str(v).strip())
    except ValueError:
        return None
    return x if 1 <= x <= 5 else None


def _yn(v: str) -> bool | None:
    s = str(v).strip().lower()
    return True if s in ("y", "yes") else False if s in ("n", "no") else None


def score_sheets(paths: list[str | Path], key_path: str | Path | None = None) -> dict[str, Any]:
    """Aggregate filled sheets (one CSV per rater). Reports distributions, not just means, and
    inter-rater agreement when two or more raters rated the same items."""
    by_rater: dict[str, dict[str, dict[str, str]]] = {}
    for p in paths:
        with Path(p).open(encoding="utf-8-sig", newline="") as fh:
            by_rater[Path(p).stem] = {r["item"]: r for r in csv.DictReader(fh)}
    out: dict[str, Any] = {"raters": sorted(by_rater), "scales": {}}
    for col in ("adequacy_1_5", "fluency_1_5", "naturalness_1_5"):
        vals = [x for rr in by_rater.values() for r in rr.values() if (x := _num(r.get(col, ""))) is not None]
        if vals:
            out["scales"][col] = {"n": len(vals), "mean": round(statistics.mean(vals), 2),
                                  "median": statistics.median(vals),
                                  "distribution": {str(k): sum(1 for v in vals if v == k) for k in range(1, 6)},
                                  "share_4_or_5": round(sum(v >= 4 for v in vals) / len(vals), 3)}
    for col in ("names_ok_y_n_na", "numbers_ok_y_n_na", "terminology_ok_y_n_na"):
        vals = [x for rr in by_rater.values() for r in rr.values() if (x := _yn(r.get(col, ""))) is not None]
        if vals:
            out["scales"][col] = {"n": len(vals), "share_ok": round(sum(vals) / len(vals), 3)}
    sev = [s for rr in by_rater.values() for r in rr.values()
           if (s := str(r.get("error_severity_none_minor_major_critical", "")).strip().lower()) in SEVERITY]
    if sev:
        out["scales"]["error_severity"] = {k: round(sev.count(k) / len(sev), 3) for k in SEVERITY} | {"n": len(sev)}
    # agreement on adequacy between rater pairs
    raters = sorted(by_rater)
    pairs = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            a, b = by_rater[raters[i]], by_rater[raters[j]]
            both = [(x, y) for it in set(a) & set(b)
                    if (x := _num(a[it].get("adequacy_1_5", ""))) is not None
                    and (y := _num(b[it].get("adequacy_1_5", ""))) is not None]
            if both:
                pairs.append({"raters": [raters[i], raters[j]], "items": len(both),
                              "exact_agreement": round(sum(x == y for x, y in both) / len(both), 3),
                              "within_1": round(sum(abs(x - y) <= 1 for x, y in both) / len(both), 3),
                              "mean_abs_diff": round(statistics.mean(abs(x - y) for x, y in both), 2)})
    if pairs:
        out["adequacy_agreement"] = pairs
    if key_path and Path(key_path).is_file():
        with Path(key_path).open(encoding="utf-8-sig", newline="") as fh:
            key = {r["item"]: r for r in csv.DictReader(fh)}
        groups: dict[str, list[float]] = defaultdict(list)
        for rr in by_rater.values():
            for it, r in rr.items():
                ade = _num(r.get("adequacy_1_5", ""))
                w = key.get(it, {}).get("asr_wer")
                if ade is None or w in (None, ""):
                    continue
                w = float(w)
                groups["ASR exact" if w == 0 else "ASR WER <= 0.2" if w <= 0.2 else "ASR WER > 0.2"].append(ade)
        out["adequacy_by_asr_quality"] = {g: {"n": len(v), "mean": round(statistics.mean(v), 2)}
                                          for g, v in sorted(groups.items())}
    return out
