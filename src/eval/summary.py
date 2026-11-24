"""One flat row per evaluation run, for comparison tables (notebooks, papers, CSV).

Reads the ``results.json`` written by ``evaluate speech | latency | mt`` and keeps the headline
numbers with their confidence intervals. Missing values stay ``None``; nothing is filled in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _get(d: Any, *path: str) -> Any:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _ms_to_s(v: Any) -> float | None:
    return round(v / 1000, 2) if isinstance(v, (int, float)) else None


def _ci(ci: Any) -> str | None:
    if isinstance(ci, dict) and "low" in ci and "high" in ci:       # bootstrap_ratio
        return f"{ci['low']}-{ci['high']}"
    return f"{ci[0]}-{ci[1]}" if isinstance(ci, (list, tuple)) and len(ci) == 2 else None


def load_results(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "results.json").read_text(encoding="utf-8"))


def _meta_row(meta: dict[str, Any], run_dir: str | Path | None) -> dict[str, Any]:
    models = meta.get("models") or {}
    return {"run": Path(run_dir).name if run_dir else None, "profile": meta.get("profile"),
            "device": meta.get("device"), "asr_model": models.get("asr"), "asr_beam": meta.get("asr_beam"),
            "mt_model": models.get("mt"), "mt_beam": meta.get("mt_beam")}


def speech_rows(results: dict[str, Any], run_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """One row per audio condition."""
    base = _meta_row(results.get("meta", {}), run_dir)
    rows = []
    for cond, c in (results.get("conditions") or {}).items():
        a, s = c.get("asr", {}), c.get("speech_to_english", {})
        rows.append({
            **base, "condition": cond, "n": c.get("n"),
            "wer": a.get("wer"), "wer_ci95": _ci(a.get("wer_ci95")),
            "wer_numbers_as_digits": _get(a, "number_words_normalized", "wer"), "cer": a.get("cer"),
            "empty_rate": a.get("empty_rate"), "decode_p50_s": _ms_to_s(_get(a, "decode_ms", "median")),
            "decode_p95_s": _ms_to_s(_get(a, "decode_ms", "p95")), "decode_p99_s": _ms_to_s(_get(a, "decode_ms", "p99")),
            "rtf": a.get("rtf"), "s2t_bleu": s.get("bleu"), "s2t_chrf": s.get("chrf"),
            "s2t_chrf_ci95": _ci(_get(s, "ci95", "chrf")), "en_number_recall": _get(s, "number_recall", "recall"),
            "oracle_chrf": _get(c, "oracle_mt", "chrf"), "asr_chrf_loss": c.get("asr_induced_chrf_loss"),
            "errors": c.get("errors"),
        })
    return rows


def mt_row(results: dict[str, Any], run_dir: str | Path | None = None) -> dict[str, Any]:
    m = results.get("mt", {})
    return {**_meta_row(results.get("meta", {}), run_dir), "n": _get(m, "latency_ms", "n"),
            "bleu": m.get("bleu"), "bleu_ci95": _ci(_get(m, "ci95", "bleu")), "chrf": m.get("chrf"),
            "chrf_ci95": _ci(_get(m, "ci95", "chrf")), "chrf_pp": m.get("chrf_pp"),
            "numbers_kept": _get(m, "numbers_kept_from_source", "recall"),
            "names_vs_ref": _get(m, "name_recall_vs_reference", "recall"),
            "omission_rate": m.get("omission_rate"), "addition_rate": m.get("addition_rate"),
            "latency_p50_s": _ms_to_s(_get(m, "latency_ms", "median")),
            "latency_p95_s": _ms_to_s(_get(m, "latency_ms", "p95")),
            "latency_p99_s": _ms_to_s(_get(m, "latency_ms", "p99"))}


def latency_row(results: dict[str, Any], run_dir: str | Path | None = None) -> dict[str, Any]:
    lat = results.get("latency", {})
    row = {**_meta_row(results.get("meta", {}), run_dir), "recordings": lat.get("n_recordings"),
           "units": lat.get("units"), "audio_s": lat.get("audio_s")}
    for stage, key in (("sw_text", "asr_final"), ("en_text", "mt_final"), ("first_audio", "first_audio")):
        s = _get(lat, "latency_from_speech_end_ms", key) or {}
        for q, name in (("median", "p50"), ("p95", "p95"), ("p99", "p99")):
            row[f"{stage}_{name}_s"] = _ms_to_s(s.get(q))
    row.update({"late_blocks": lat.get("push_blocks_late"), "rss_peak_mb": lat.get("rss_peak_mb"),
                "errors": len(lat.get("errors") or []) if isinstance(lat.get("errors"), list) else lat.get("errors")})
    return row


def summarize_run(run_dir: str | Path) -> list[dict[str, Any]]:
    """Rows for any evaluation run directory (kind inferred from results.json)."""
    res = load_results(run_dir)
    if "conditions" in res:
        return speech_rows(res, run_dir)
    if "mt" in res and "latency" not in res:
        return [mt_row(res, run_dir)]
    if "latency" in res and "speech" not in res:
        return [latency_row(res, run_dir)]
    rows = speech_rows({"meta": res.get("meta", {}), "conditions": res.get("speech", {})}, run_dir)
    rows.append(latency_row(res, run_dir))
    rows.append(mt_row(res, run_dir))
    return rows
