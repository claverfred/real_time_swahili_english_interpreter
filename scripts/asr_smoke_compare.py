"""Compare ASR candidates on the synthetic Swahili smoke set (integration-level evidence only).

    python scripts/asr_smoke_compare.py --profile cpu --models base small large-v3-turbo

Reports WER/CER (normalization disclosed), per-clip decode latency (mean/median/p95/p99), RTF,
download/load/cold/warm timings, and the resolved model revision. Synthetic TTS speech is much
easier than real speech: use this to choose what to benchmark properly, not as a final result.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import create_asr  # noqa: E402
from src.audio import load_audio  # noqa: E402
from src.config import load_config  # noqa: E402
from src.eval.metrics import wer_cer  # noqa: E402
from src.platform.resolve import resolve_runtime  # noqa: E402
from src.platform.storage import prepare_storage  # noqa: E402
from src.utils.manifest import build_manifest, new_run_dir, write_manifest  # noqa: E402
from src.utils.stats import summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cpu")
    ap.add_argument("--models", nargs="+", default=["base", "small"])
    ap.add_argument("--data", default="data/synthetic/sw_tts_v1")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    clips = [(it, load_audio(data / it["audio"])[0]) for it in manifest["items"]]
    rows = []
    cfg0, runtime, fb = resolve_runtime(load_config(args.profile, args.overrides))
    paths = prepare_storage(cfg0)
    proc = psutil.Process()

    for model in args.models:
        cfg = cfg0.model_copy(deep=True)
        cfg.asr.model = model
        cfg.asr.fallback_models = []
        gc.collect()
        rss0 = proc.memory_info().rss
        print(f"== {model}", file=sys.stderr)
        try:
            asr = create_asr(cfg, fb)
        except Exception as exc:  # failed candidates are reported, never hidden
            rows.append({"model": model, "status": "failed", "error": str(exc)})
            print(f"   FAILED: {exc}", file=sys.stderr)
            continue
        timing = dict(asr.load_info)
        timing["cold_ms"] = round(asr.warmup(), 1)
        timing["warm_ms"] = round(asr.warmup(), 1)
        refs, hyps, lat, per_clip = [], [], [], []
        t_audio = 0.0
        for it, audio in clips:
            t0 = time.perf_counter()
            res = asr.transcribe(audio)
            ms = (time.perf_counter() - t0) * 1000
            refs.append(it["text"])
            hyps.append(res.text)
            lat.append(ms)
            t_audio += res.duration_s
            per_clip.append({"id": it["id"], "ref": it["text"], "hyp": res.text, "ms": round(ms, 1),
                             "uncertain": res.uncertain,
                             "rejected": [s.rejected for s in res.rejected_segments]})
            print(f"   {it['id']:<28} {ms:7.0f} ms  {res.text}", file=sys.stderr)
        rss_mb = (proc.memory_info().rss - rss0) / 1e6
        acc = wer_cer(refs, hyps)
        by_voice = {v: wer_cer([r for r, c in zip(refs, clips, strict=True) if c[0]["voice"] == v],
                               [h for h, c in zip(hyps, clips, strict=True) if c[0]["voice"] == v])["wer"]
                    for v in manifest["voices"]}
        rows.append({"model": model, "status": "ok", "accuracy": acc, "wer_by_voice": by_voice,
                     "latency_ms": summarize(lat), "rtf": round(sum(lat) / 1000 / t_audio, 4),
                     "rss_delta_mb": round(rss_mb, 1), "timing": timing, "clips": per_clip})
        del asr
        gc.collect()

    run_dir = new_run_dir(paths["output_dir"], "asr-smoke")
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(run_dir, build_manifest(
        run_id=run_dir.name, mode="asr-smoke-compare", config=cfg0.model_dump(mode="json", by_alias=True),
        runtime=runtime.to_dict(), fallbacks=fb,
        datasets=[{"name": manifest["name"], "kind": manifest["kind"], "warning": manifest["warning"],
                   "items": len(clips)}]))

    print(f"\n{'model':<16}{'WER':>7}{'CER':>7}{'p50 ms':>9}{'p95 ms':>9}{'RTF':>7}{'load ms':>9}{'RSS MB':>8}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['model']:<16} FAILED  {r['error'][:60]}")
            continue
        lt = r["latency_ms"]
        print(f"{r['model']:<16}{r['accuracy']['wer']:>7.3f}{r['accuracy']['cer']:>7.3f}{lt['median']:>9.0f}"
              f"{lt['p95']:>9.0f}{r['rtf']:>7.3f}{r['timing']['load_ms']:>9.0f}{r['rss_delta_mb']:>8.0f}")
    print(f"\n(synthetic TTS speech — smoke evidence only)  -> {run_dir}")


if __name__ == "__main__":
    main()
