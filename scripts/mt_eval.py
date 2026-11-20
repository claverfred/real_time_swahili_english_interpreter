"""Evaluate MT candidates on FLORES-200 Swahili->English against human references.

    python scripts/fetch_flores200.py
    python scripts/mt_eval.py --profile cpu --models facebook/nllb-200-distilled-600M Helsinki-NLP/opus-mt-swc-en --n 200

Reports BLEU, chrF and chrF++ (sacrebleu, signatures recorded), per-sentence latency at
batch size 1 (the live-pipeline case: mean/median/p95/p99), number preservation, flag
rates, RAM delta and load timings. ``--context`` adds a run with the previous sentence as
context (FLORES devtest is ordered by source article, so neighbours are usually related).
References are human translations; hypotheses are never used as references.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import psutil
import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.mt import create_translator  # noqa: E402
from src.platform.resolve import resolve_runtime  # noqa: E402
from src.platform.storage import prepare_storage  # noqa: E402
from src.text.entities import check_preservation  # noqa: E402
from src.utils.manifest import build_manifest, new_run_dir, write_manifest  # noqa: E402
from src.utils.stats import summarize  # noqa: E402


def scores(hyps: list[str], refs: list[str]) -> dict[str, object]:
    bleu = sacrebleu.metrics.BLEU()
    chrf = sacrebleu.metrics.CHRF()
    chrfpp = sacrebleu.metrics.CHRF(word_order=2)
    b, c, cp = bleu.corpus_score(hyps, [refs]), chrf.corpus_score(hyps, [refs]), chrfpp.corpus_score(hyps, [refs])
    return {"bleu": round(b.score, 2), "chrf": round(c.score, 2), "chrf_pp": round(cp.score, 2),
            "signatures": {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature()),
                           "chrf_pp": str(chrfpp.get_signature())}}


def run_variant(tr, srcs: list[str], refs: list[str], context: bool) -> dict[str, object]:
    tr.cfg.context_mode = "prefix" if context else "none"
    tr._cache.clear()
    hyps, lat, flags, num_src, num_missing = [], [], {}, 0, 0
    history: list[tuple[str, str]] = []
    for s in srcs:
        t0 = time.perf_counter()
        r = tr.translate(s, context=history if context else None)
        lat.append((time.perf_counter() - t0) * 1000)
        hyps.append(r.text)
        history = (history + [(s, r.text)])[-1:]
        for f in r.flags:
            flags[f] = flags.get(f, 0) + 1
        p = check_preservation(s, r.text)["numbers"]
        num_src += len(p["source"])
        num_missing += len(p["missing"])
    n = len(srcs)
    return {**scores(hyps, refs), "latency_ms": summarize(lat),
            "flag_rates": {k: round(v / n, 4) for k, v in sorted(flags.items())},
            "number_preservation": round(1 - num_missing / num_src, 4) if num_src else None,
            "numbers_in_source": num_src,
            "samples": [{"src": s, "ref": r, "hyp": h} for s, r, h in list(zip(srcs, refs, hyps, strict=True))[:15]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cpu")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="data/flores200")
    ap.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--n", type=int, default=200, help="first N sentences (0 = all)")
    ap.add_argument("--beam", type=int)
    ap.add_argument("--engine")
    ap.add_argument("--context", action="store_true", help="also evaluate with 1 sentence of context")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    data = Path(args.data)
    srcs = (data / f"{args.split}.swh_Latn.txt").read_text(encoding="utf-8").splitlines()
    refs = (data / f"{args.split}.eng_Latn.txt").read_text(encoding="utf-8").splitlines()
    if args.n:
        srcs, refs = srcs[:args.n], refs[:args.n]
    overrides = list(args.overrides) + ([f"mt.beam_size={args.beam}"] if args.beam else []) \
        + ([f"mt.engine={args.engine}"] if args.engine else [])
    cfg0, runtime, fb = resolve_runtime(load_config(args.profile, overrides))
    paths = prepare_storage(cfg0)
    proc = psutil.Process()
    rows = []
    for model in args.models:
        cfg = cfg0.model_copy(deep=True)
        cfg.mt.model, cfg.mt.fallback_models, cfg.mt.revision = model, [], "main"
        from src.mt.langs import family_of
        cfg.mt.backend = family_of(model, cfg.mt.backend)
        gc.collect()
        rss0 = proc.memory_info().rss
        print(f"== {model}", file=sys.stderr)
        try:
            tr = create_translator(cfg, fb)
        except Exception as exc:
            rows.append({"model": model, "status": "failed", "error": str(exc)})
            print(f"   FAILED: {exc}", file=sys.stderr)
            continue
        timing = dict(tr.load_info)
        timing["cold_ms"] = round(tr.warmup(), 1)
        timing["warm_ms"] = round(tr.warmup(), 1)
        row = {"model": model, "status": "ok", "beam": cfg.mt.beam_size, "timing": timing,
               "no_context": run_variant(tr, srcs, refs, context=False)}
        print(f"   no-context chrF {row['no_context']['chrf']}  BLEU {row['no_context']['bleu']}", file=sys.stderr)
        if args.context:
            row["context_1"] = run_variant(tr, srcs, refs, context=True)
            print(f"   context    chrF {row['context_1']['chrf']}  BLEU {row['context_1']['bleu']}", file=sys.stderr)
        row["rss_delta_mb"] = round((proc.memory_info().rss - rss0) / 1e6, 1)
        rows.append(row)
        del tr
        gc.collect()

    run_dir = new_run_dir(paths["output_dir"], "mt-eval")
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    write_manifest(run_dir, build_manifest(
        run_id=run_dir.name, mode="mt-eval", config=cfg0.model_dump(mode="json", by_alias=True),
        runtime=runtime.to_dict(), fallbacks=fb,
        datasets=[{"name": "flores200", "split": args.split, "items": len(srcs), "subset": "first N",
                   "archive_sha256": manifest.get("archive_sha256"), "license": manifest.get("license")}]))

    print(f"\nFLORES-200 {args.split}, first {len(srcs)} sentences, swh_Latn -> eng_Latn")
    print(f"{'model':<36}{'variant':<12}{'BLEU':>7}{'chrF':>7}{'chrF++':>8}{'p50 ms':>8}{'p95 ms':>8}{'num%':>7}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['model']:<36}FAILED {r['error'][:70]}")
            continue
        for v in ("no_context", "context_1"):
            if v in r:
                m = r[v]
                num = f"{m['number_preservation'] * 100:.0f}" if m["number_preservation"] is not None else "-"
                print(f"{r['model']:<36}{v:<12}{m['bleu']:>7.1f}{m['chrf']:>7.1f}{m['chrf_pp']:>8.1f}"
                      f"{m['latency_ms']['median']:>8.0f}{m['latency_ms']['p95']:>8.0f}{num:>7}")
    print(f"\n-> {run_dir}")


if __name__ == "__main__":
    main()
