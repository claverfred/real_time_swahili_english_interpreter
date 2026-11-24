"""Speech evaluations through the deployed pipeline (spec sections 12-14).

* :func:`run_speech` - each recording goes through the same RealtimeSwahiliInterpreter used live
  (VAD -> ASR -> correction list -> stabilization -> sentence aggregation -> MT), one utterance at
  a time with a context reset in between. It yields Swahili ASR metrics *and* speech-to-English
  metrics against human references, plus an "oracle" translation of the human Swahili transcript,
  so translation loss caused by ASR errors can be separated from MT's own errors.
* :func:`run_latency` - recordings joined with pauses and streamed at real-time pace with TTS on,
  for listener-relevant latency (from the last speech frame) with p50/p95/p99.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from ..utils.stats import summarize
from . import metrics as M
from .datasets import SpeechItem
from .perturb import apply as perturb

SR = 16000
Progress = Callable[[int, int, dict[str, Any]], None]


def _load(item: SpeechItem, condition: str) -> np.ndarray:
    from ..audio import load_audio

    audio, _ = load_audio(item.audio_path, SR)
    return perturb(condition, audio, item.item_id)


async def run_speech(cfg: Any, items: list[SpeechItem], components: Any, *, condition: str = "clean",
                     oracle: bool = True, progress: Progress | None = None) -> list[dict[str, Any]]:
    """Per-item results for one audio condition (batch speed; latencies here are compute times)."""
    from ..pipeline import EventType, RealtimeSwahiliInterpreter

    interp = RealtimeSwahiliInterpreter(cfg, components=components, enable_tts=False, partials=False,
                                        local_files_only=True)
    await interp.start()
    current: dict[str, Any] = {}

    async def collect() -> None:
        async for ev in interp.events():
            if ev.type == EventType.ASR_FINAL:
                current.setdefault("asr", []).append(ev)
            elif ev.type == EventType.ERROR:
                current.setdefault("errors", []).append(str(ev.data.get("error") or ev.data)[:300])

    task = asyncio.create_task(collect())
    rows: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    try:
        for k, item in enumerate(items):
            audio = await loop.run_in_executor(None, _load, item, condition)
            current.clear()
            n_units = len(interp.units())
            t0 = time.perf_counter()
            for i in range(0, len(audio), SR):
                await interp.push_audio(audio[i:i + SR], SR)
            await interp.flush()
            await asyncio.sleep(0)                      # let the collector drain the last events
            wall = time.perf_counter() - t0
            units = interp.units()[n_units:]
            asr_evs = list(current.get("asr", []))
            row = {
                "item_id": item.item_id, "condition": condition, "audio_s": round(len(audio) / SR, 3),
                "sw_ref": item.sw_ref, "en_ref": item.en_ref, **{k2: v for k2, v in item.meta.items()},
                "sw_hyp": " ".join(e.text or "" for e in asr_evs).strip(),
                "sw_hyp_uncorrected": " ".join(e.data.get("original", e.text) or "" for e in asr_evs).strip(),
                "corrections": sum(int(e.data.get("corrections", 0) or 0) for e in asr_evs),
                "segments": len(asr_evs),
                "asr_ms": round(sum(float(e.data.get("latency_ms") or 0) for e in asr_evs), 1),
                "rejected": [r for e in asr_evs for r in (e.data.get("rejected") or [])],
                "uncertain": any(bool(e.data.get("uncertain")) for e in asr_evs),
                "en_hyp": " ".join(u["english"] for u in units if u.get("english")).strip(),
                "mt_ms": round(sum(float(u.get("mt_ms") or 0) for u in units), 1),
                "mt_flags": sorted({f for u in units for f in u.get("flags", [])}),
                "wall_ms": round(wall * 1000, 1), "errors": list(current.get("errors", [])),
            }
            if oracle and item.en_ref is not None:
                tr = components.translator
                res = await loop.run_in_executor(None, tr.translate, item.sw_ref)
                row["en_oracle"] = res.text
            rows.append(row)
            await interp.reset()
            if progress:
                progress(k + 1, len(items), row)
    finally:
        await interp.shutdown(drain=False)
        await task
    return rows


def score_speech(rows: list[dict[str, Any]], *, bootstrap: int = 1000, failures: int = 12) -> dict[str, Any]:
    """Aggregate metrics for one condition. Never only an average: CIs, percentiles, breakdowns."""
    refs = [r["sw_ref"] for r in rows]
    hyps = [r["sw_hyp"] for r in rows]
    per = [M.utterance_errors(r["sw_ref"], r["sw_hyp"]) for r in rows]
    for r, p in zip(rows, per, strict=True):
        r["wer"] = round(p["word_errors"] / p["ref_words"], 4) if p["ref_words"] else None
    audio_s = sum(r["audio_s"] for r in rows)
    asr: dict[str, Any] = {
        **M.wer_cer(refs, hyps),
        "wer_ci95": M.bootstrap_ratio([p["word_errors"] for p in per], [p["ref_words"] for p in per], n_boot=bootstrap),
        "cer_ci95": M.bootstrap_ratio([p["char_errors"] for p in per], [p["ref_chars"] for p in per], n_boot=bootstrap),
        "wer_uncorrected": M.wer_cer(refs, [r["sw_hyp_uncorrected"] for r in rows])["wer"],
        "number_words_normalized": {
            **{k: v for k, v in M.wer_cer([M.sw_numbers_to_digits(M.normalize(x)) for x in refs],
                                          [M.sw_numbers_to_digits(M.normalize(x)) for x in hyps]).items()
               if k in ("wer", "cer")},
            "method": M.NUMBER_NORMALIZATION},
        "utterance_wer": summarize([r["wer"] for r in rows if r["wer"] is not None]),
        "sentences_exact": round(sum(1 for r in rows if r["wer"] == 0) / len(rows), 4),
        "number_recall": M.recall(refs, [M.sw_numbers_to_digits(M.normalize(h)) for h in hyps], "numbers"),
        "name_recall": M.recall(refs, hyps, "names"),
        "empty_rate": round(sum(1 for h in hyps if not h.strip()) / len(rows), 4),
        "rejected_segment_rate": round(sum(1 for r in rows if r["rejected"]) / len(rows), 4),
        "rejected_reasons": dict(Counter(x for r in rows for x in r["rejected"])),
        "uncertain_rate": round(sum(1 for r in rows if r["uncertain"]) / len(rows), 4),
        "corrections_applied": sum(r["corrections"] for r in rows),
        "multi_segment_rate": round(sum(1 for r in rows if r["segments"] > 1) / len(rows), 4),
        "decode_ms": summarize([r["asr_ms"] for r in rows]),
        "rtf": round(sum(r["asr_ms"] for r in rows) / 1000 / audio_s, 3) if audio_s else None,
    }
    by_gender: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_gender[str(r.get("gender") or "unknown")].append(r)
    if len(by_gender) > 1:
        asr["by_gender"] = {g: {"n": len(v), "wer": M.wer_cer([x["sw_ref"] for x in v], [x["sw_hyp"] for x in v])["wer"]}
                            for g, v in sorted(by_gender.items())}
    out: dict[str, Any] = {"n": len(rows), "audio_s": round(audio_s, 1), "asr": asr,
                           "errors": sum(len(r["errors"]) for r in rows)}

    paired = [r for r in rows if r.get("en_ref") and r.get("en_hyp") is not None]
    if paired:
        en_refs = [r["en_ref"] for r in paired]
        en_hyps = [r["en_hyp"] for r in paired]
        for r in paired:
            r["chrf"] = M.sentence_chrf(r["en_hyp"], r["en_ref"])
        s2t: dict[str, Any] = {
            **M.mt_scores(en_hyps, en_refs, bootstrap=bootstrap // 4),
            "number_recall": M.recall(en_refs, en_hyps, "numbers"),
            "name_recall": M.recall(en_refs, en_hyps, "names"),
            "acronym_recall": M.recall(en_refs, en_hyps, "acronyms"),
            **M.length_ratio_rates(en_refs, en_hyps),
            "mt_ms": summarize([r["mt_ms"] for r in paired]),
            "flag_rates": {k: round(v / len(paired), 4) for k, v in
                           sorted(Counter(f for r in paired for f in r["mt_flags"]).items())},
        }
        buckets = [("0 (exact)", 0, 0), ("0-0.2", 1e-9, 0.2), ("0.2-0.5", 0.2 + 1e-9, 0.5), (">0.5", 0.5 + 1e-9, 99)]
        s2t["chrf_by_asr_wer"] = {
            name: {"n": len(b), "chrf_mean": round(float(np.mean([r["chrf"] for r in b])), 1)}
            for name, lo, hi in buckets if (b := [r for r in paired if r["wer"] is not None and lo <= r["wer"] <= hi])}
        out["speech_to_english"] = s2t
        oracle = [r for r in paired if r.get("en_oracle") is not None]
        if oracle:
            out["oracle_mt"] = {
                **M.mt_scores([r["en_oracle"] for r in oracle], [r["en_ref"] for r in oracle]),
                "note": "NLLB translating the human Swahili transcript of the same recordings: the "
                        "translation quality if ASR were perfect"}
            out["asr_induced_chrf_loss"] = round(out["oracle_mt"]["chrf"] - s2t["chrf"], 2)

    worst = sorted((r for r in rows if r["wer"] is not None), key=lambda r: -r["wer"])[:failures]
    out["failure_examples"] = [{k: r.get(k) for k in ("item_id", "wer", "sw_ref", "sw_hyp", "en_ref", "en_hyp", "chrf")}
                               for r in worst]
    return out


# -- real-time latency ---------------------------------------------------------------------------
def build_stream(items: list[SpeechItem], gap_s: float, lead_s: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Recordings joined by ``gap_s`` of silence (the speaker pausing between sentences)."""
    from ..audio import load_audio

    parts = [np.zeros(int(lead_s * SR), np.float32)]
    spans, t = [], lead_s
    for it in items:
        a, _ = load_audio(it.audio_path, SR)
        parts += [a.astype(np.float32), np.zeros(int(gap_s * SR), np.float32)]
        spans.append({"item_id": it.item_id, "start_s": round(t, 3), "end_s": round(t + len(a) / SR, 3)})
        t += len(a) / SR + gap_s
    return np.concatenate(parts), spans


async def run_latency(cfg: Any, items: list[SpeechItem], components: Any, *, gap_s: float = 2.0,
                      block_s: float = 0.1, tts: bool = True, progress: Progress | None = None) -> dict[str, Any]:
    """Stream real speech at real-time pace and report listener latency (from the last speech
    frame) per stage with p50/p95/p99, queue waits, RTF, RAM and dropped/failed counts."""
    import psutil

    from ..pipeline import EventType, RealtimeSwahiliInterpreter

    audio, spans = await asyncio.get_running_loop().run_in_executor(None, build_stream, items, gap_s)
    interp = RealtimeSwahiliInterpreter(cfg, components=components, enable_tts=tts, partials=None,
                                        local_files_only=True)
    await interp.start()
    proc, rss_peak = psutil.Process(), 0
    done = 0

    async def collect() -> None:
        nonlocal done
        async for ev in interp.events():
            if ev.type == EventType.PIPELINE_COMPLETE:
                done += 1
                if progress:
                    progress(done, len(items), {"english": ev.text})

    task = asyncio.create_task(collect())
    step = int(block_s * SR)
    t0 = time.perf_counter()
    late_blocks = 0
    for k, i in enumerate(range(0, len(audio), step)):
        await interp.push_audio(audio[i:i + step], SR)
        due = t0 + (k + 1) * block_s
        lag = due - time.perf_counter()
        if lag > 0:
            await asyncio.sleep(lag)
        elif lag < -block_s:
            late_blocks += 1
        if k % 10 == 0:
            rss_peak = max(rss_peak, proc.memory_info().rss)
    summary = await interp.shutdown(drain=True)
    await task
    rss_peak = max(rss_peak, proc.memory_info().rss)
    units = interp.units()
    lat = summary["latency_from_speech_end_ms"]
    return {
        "n_recordings": len(items), "units": len(units), "audio_s": round(len(audio) / SR, 1), "gap_s": gap_s,
        "pace": f"real time, {int(block_s * 1000)} ms blocks", "tts": tts,
        "latency_from_speech_end_ms": lat, "stage_ms": summary.get("stage_ms"),
        "queue_wait_ms": summary.get("queue_wait_ms"), "wall_s": summary.get("wall_s"),
        "push_blocks_late": late_blocks, "rss_peak_mb": round(rss_peak / 2**20, 1),
        "counters": summary.get("counters"), "errors": summary.get("errors"),
        "units_detail": [{k: u.get(k) for k in ("swahili", "english", "start_s", "end_s", "lat_asr_final_ms",
                                                 "lat_mt_final_ms", "lat_first_audio_ms", "lat_full_audio_ms")}
                         for u in units],
        "spans": spans,
    }


# -- text-only MT ---------------------------------------------------------------------------------
def run_mt(translator: Any, sources: list[str], references: list[str], *, bootstrap: int = 250,
           progress: Progress | None = None) -> dict[str, Any]:
    """Text MT on human references: quality, preservation, length proxies, latency p50/p95/p99."""
    from ..text.entities import check_preservation

    hyps, lat, flags = [], [], Counter()
    cache = getattr(translator, "_cache", None)
    if cache is not None and hasattr(cache, "clear"):
        cache.clear()
    for k, s in enumerate(sources):
        t0 = time.perf_counter()
        r = translator.translate(s)
        lat.append((time.perf_counter() - t0) * 1000)
        hyps.append(r.text)
        flags.update(r.flags)
        if progress:
            progress(k + 1, len(sources), {"english": r.text})
    src_numbers = sum(len(check_preservation(s, h)["numbers"]["source"]) for s, h in zip(sources, hyps, strict=True))
    return {
        **M.mt_scores(hyps, references, bootstrap=bootstrap),
        "number_recall_vs_reference": M.recall(references, hyps, "numbers"),
        "name_recall_vs_reference": M.recall(references, hyps, "names"),
        "acronym_recall_vs_reference": M.recall(references, hyps, "acronyms"),
        "numbers_kept_from_source": M.recall(sources, hyps, "numbers"),
        "source_numbers": src_numbers,
        **M.length_ratio_rates(references, hyps),
        "latency_ms": summarize(lat),
        "flag_rates": {k: round(v / len(sources), 4) for k, v in sorted(flags.items())},
        "samples": [{"src": s, "ref": r, "hyp": h} for s, r, h in list(zip(sources, references, hyps, strict=True))[:20]],
        "hypotheses": hyps,
    }
