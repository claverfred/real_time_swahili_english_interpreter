"""Compare Whisper models on real Swahili speech (FLEURS sw_ke test), CPU, pipeline decode settings.

    python scripts/fetch_fleurs_sw.py
    python scripts/asr_eval_fleurs.py --n 60 --models small alphaedge-ai/whisper-small-swh-32768

Each model is loaded through the pipeline's own backend (so converted fine-tunes are exercised
exactly as `interpret` will use them). Reports WER/CER (normalization disclosed), decode latency
p50/p95, RTF, load time and RSS. Results: results/asr_eval/<run>/results.json + hypotheses.jsonl.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from src.asr.faster_whisper_backend import AsrError, FasterWhisperAsr  # noqa: E402
from src.audio import load_audio  # noqa: E402
from src.config.schema import AsrConfig  # noqa: E402
from src.eval.metrics import wer_cer  # noqa: E402
from src.platform.cpu import auto_threads  # noqa: E402
from src.utils.manifest import new_run_dir  # noqa: E402
from src.utils.stats import summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="data/fleurs_sw/test.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--beam", type=int, default=1, help="cpu profile uses 1")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--out", default="results/asr_eval")
    args = ap.parse_args()

    items = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()][: args.n]
    clips = [load_audio(it["path"], 16000)[0] for it in items]
    total_audio = sum(len(c) for c in clips) / 16000
    run_dir = new_run_dir(Path(args.out), "fleurs")
    threads = auto_threads(0)
    results, hyp_rows = [], []
    print(f"{len(items)} utterances, {total_audio:.0f} s audio, {threads} threads -> {run_dir}", flush=True)

    for model in args.models:
        cfg = AsrConfig(model=model, beam_size=args.beam, compute_type=args.compute_type)
        asr = FasterWhisperAsr(cfg, compute_type=args.compute_type, cpu_threads=threads, cache_dir="cache")
        row: dict[str, object] = {"model": model}
        try:
            t0 = time.perf_counter()
            row["load"] = asr.load()
            row["load_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            asr.warmup()
        except AsrError as exc:
            row["status"] = f"failed: {exc}"
            print(f"{model}: {row['status']}", flush=True)
            results.append(row)
            continue
        hyps, ms = [], []
        for it, clip in zip(items, clips, strict=True):
            res = asr.transcribe(clip)
            hyps.append(res.text)
            ms.append(res.inference_ms)
            hyp_rows.append({"model": model, "id": it["file"], "ref": it["raw"], "hyp": res.text,
                             "ms": res.inference_ms})
        acc = wer_cer([it["raw"] for it in items], hyps)
        row.update(status="ok", accuracy=acc, decode_ms=summarize(ms),
                   rtf=round(sum(ms) / 1000 / total_audio, 3),
                   rss_mb=round(psutil.Process().memory_info().rss / 2**20, 1))
        results.append(row)
        print(f"{model:<58} WER {acc['wer']:.3f}  CER {acc['cer']:.3f}  decode p50 {np.median(ms):6.0f} ms  "
              f"RTF {row['rtf']:.3f}", flush=True)
        asr.unload()
        del asr
        gc.collect()

    meta = {"data": args.data, "n": len(items), "audio_s": round(total_audio, 1), "beam": args.beam,
            "compute_type": args.compute_type, "threads": threads}
    (run_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2,
                                                     ensure_ascii=False, default=str), encoding="utf-8")
    (run_dir / "hypotheses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hyp_rows),
                                              encoding="utf-8")
    print("\nsummary")
    for r in results:
        if r.get("status") == "ok":
            print(f"  {r['model']:<58} WER {r['accuracy']['wer']:.3f}  RTF {r['rtf']:.3f}  "
                  f"p50 {r['decode_ms']['median']:.0f} ms")
        else:
            print(f"  {r['model']:<58} {r['status']}")


if __name__ == "__main__":
    main()
