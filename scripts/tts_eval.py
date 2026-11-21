"""Compare TTS backends/voices: latency, RTF, memory, level checks and an intelligibility proxy.

    python scripts/tts_eval.py --profile cpu --candidates piper:en_US-joe-medium piper:en_US-lessac-medium sapi:default

Text: the first N English FLORES-200 devtest references (natural news sentences) plus a fixed
set of names, numbers, acronyms and questions.

Intelligibility proxy: each clip is transcribed back with Whisper (``--asr-model``, English)
and scored by WER/CER against the input text (normalization disclosed by wer_cer). This is a
machine proxy for intelligibility, not a listening test. Naturalness and prosody still need
human ratings (Phase 8).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr.faster_whisper_backend import FasterWhisperAsr  # noqa: E402
from src.audio.io import resample  # noqa: E402
from src.config import load_config  # noqa: E402
from src.config.schema import AsrConfig  # noqa: E402
from src.eval.metrics import wer_cer  # noqa: E402
from src.platform.resolve import resolve_runtime  # noqa: E402
from src.platform.storage import prepare_storage  # noqa: E402
from src.tts import create_synthesizer  # noqa: E402
from src.utils.manifest import build_manifest, new_run_dir, write_manifest  # noqa: E402
from src.utils.stats import summarize  # noqa: E402

SPECIAL = [
    "The price of tomatoes is 2,500 shillings per kilogram.",
    "President Samia Suluhu Hassan arrived in Dodoma on Tuesday.",
    "The train leaves Dar es Salaam at 7:30 in the morning.",
    "Are you coming to the meeting tomorrow?",
    "TANESCO said power would return by 6 p.m.",
    "About 45% of the students passed the exam in 2023.",
    "Julius Nyerere was the first president of Tanzania.",
    "Heavy rain is expected in Mbeya, Iringa and Morogoro this week.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cpu")
    ap.add_argument("--candidates", nargs="+", required=True, help="backend:voice, e.g. piper:en_US-joe-medium")
    ap.add_argument("--n", type=int, default=25, help="FLORES devtest English sentences to add")
    ap.add_argument("--flores", default="data/flores200/devtest.eng_Latn.txt")
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    texts = list(SPECIAL)
    if args.n and Path(args.flores).exists():
        texts += Path(args.flores).read_text(encoding="utf-8").splitlines()[:args.n]
    cfg0, runtime, fb = resolve_runtime(load_config(args.profile, args.overrides))
    paths = prepare_storage(cfg0)
    asr = FasterWhisperAsr(AsrConfig(model=args.asr_model, language="en", beam_size=5), model=args.asr_model,
                           device=cfg0.environment.device, compute_type=cfg0.asr.compute_type,
                           cpu_threads=cfg0.environment.cpu_threads, cache_dir=cfg0.storage.cache_dir)
    asr.load()
    proc = psutil.Process()
    rows = []
    for cand in args.candidates:
        backend, _, voice = cand.partition(":")
        cfg = cfg0.model_copy(deep=True)
        cfg.tts.backend, cfg.tts.fallback_backends = backend, []
        cfg.tts.voice = voice or cfg.tts.voices.get(backend, "default")
        if backend != cfg0.tts.backend:
            cfg.tts.revision = "main"
        gc.collect()
        rss0 = proc.memory_info().rss
        print(f"== {cand}", file=sys.stderr)
        try:
            syn = create_synthesizer(cfg, fb)
        except Exception as exc:
            rows.append({"candidate": cand, "status": "failed", "error": str(exc)})
            print(f"   FAILED: {exc}", file=sys.stderr)
            continue
        timing = dict(syn.load_info)
        timing["cold_ms"] = round(syn.warmup(), 1)
        timing["warm_ms"] = round(syn.warmup(), 1)
        first, total, audio_s, warn, hyps, clips = [], [], 0.0, 0, [], []
        for t in texts:
            res = syn.synthesize(t)
            first.append(res.first_audio_ms)
            total.append(res.total_ms)
            audio_s += res.duration_s
            warn += any(c.warnings for c in res.chunks)
            hyp = asr.transcribe(resample(res.audio, res.sample_rate, 16000)).text
            hyps.append(hyp)
            clips.append({"text": t, "asr": hyp, "first_ms": res.first_audio_ms, "total_ms": res.total_ms,
                          "audio_s": round(res.duration_s, 3), "warnings": [w for c in res.chunks for w in c.warnings]})
        rss_mb = (proc.memory_info().rss - rss0) / 1e6
        intel_all = wer_cer(texts, hyps)
        intel_special = wer_cer(texts[:len(SPECIAL)], hyps[:len(SPECIAL)])
        rows.append({"candidate": cand, "status": "ok", "license": syn.backend.license,
                     "sample_rate_native": syn.backend.sample_rate, "sample_rate_out": syn.sample_rate,
                     "first_audio_ms": summarize(first), "total_ms": summarize(total),
                     "rtf": round(sum(total) / 1000 / audio_s, 4), "audio_s": round(audio_s, 2),
                     "clips_with_warnings": warn, "intelligibility_all": intel_all,
                     "intelligibility_names_numbers": intel_special, "rss_delta_mb": round(rss_mb, 1),
                     "timing": timing, "clips": clips})
        print(f"   first p50 {summarize(first)['median']:.0f} ms  RTF {rows[-1]['rtf']:.3f}  "
              f"ASR-WER {intel_all['wer']:.3f}", file=sys.stderr)
        del syn
        gc.collect()

    run_dir = new_run_dir(paths["output_dir"], "tts-eval")
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(run_dir, build_manifest(
        run_id=run_dir.name, mode="tts-eval", config=cfg0.model_dump(mode="json", by_alias=True),
        runtime=runtime.to_dict(), fallbacks=fb,
        datasets=[{"name": "flores200 devtest eng_Latn", "items": len(texts) - len(SPECIAL)},
                  {"name": "fixed names/numbers set", "items": len(SPECIAL)}],
        extra={"intelligibility_proxy": f"faster-whisper {args.asr_model} (en) round-trip WER/CER"}))

    print(f"\n{len(texts)} sentences ({len(SPECIAL)} names/numbers + {len(texts) - len(SPECIAL)} FLORES)")
    print(f"{'candidate':<32}{'first p50':>10}{'p95':>7}{'p99':>7}{'RTF':>7}{'WER':>7}{'WER n&n':>9}{'warn':>6}{'RSS MB':>8}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['candidate']:<32} FAILED {r['error'][:60]}")
            continue
        f = r["first_audio_ms"]
        print(f"{r['candidate']:<32}{f['median']:>10.0f}{f['p95']:>7.0f}{f['p99']:>7.0f}{r['rtf']:>7.3f}"
              f"{r['intelligibility_all']['wer']:>7.3f}{r['intelligibility_names_numbers']['wer']:>9.3f}"
              f"{r['clips_with_warnings']:>6}{r['rss_delta_mb']:>8.0f}")
    print(f"\n(intelligibility = Whisper round-trip, a proxy, not a listening test)  -> {run_dir}")


if __name__ == "__main__":
    main()
