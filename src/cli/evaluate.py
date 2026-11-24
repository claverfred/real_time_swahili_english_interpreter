"""``evaluate`` and ``benchmark``: measured quality and latency against human references.

    python -m src.cli evaluate speech --dataset fleurs --profile cpu                 # all 487 FLEURS recordings
    python -m src.cli evaluate speech --n 100 --conditions clean noise_10db reverb_0.6s
    python -m src.cli evaluate latency --n 30                                         # real-time pace, TTS on
    python -m src.cli evaluate mt --dataset flores200 --n 0                           # full devtest
    python -m src.cli evaluate human-sheet --from results/eval-speech-... --n 60 --audio
    python -m src.cli evaluate human-score rater1.csv rater2.csv --key .../key.csv
    python -m src.cli benchmark --profile cpu                                          # quick suite

Each run writes results.json, per-item rows, report.md and a manifest (models, commits,
dataset provenance, config) under storage.output_dir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from ..config import load_config


def add_parser(sub: argparse._SubParsersAction, add_profile_args) -> None:  # type: ignore[type-arg]
    from ..eval.perturb import CONDITIONS

    p = sub.add_parser("evaluate", help="evaluate ASR, translation, speech-to-English and latency")
    ev = p.add_subparsers(dest="task", required=True)

    s = ev.add_parser("speech", help="Swahili speech -> ASR + English text, scored against human references")
    s.add_argument("--dataset", choices=["fleurs", "synthetic"], default="fleurs")
    s.add_argument("--n", type=int, default=0, help="first N recordings (0 = all)")
    s.add_argument("--conditions", nargs="+", default=["clean"], choices=list(CONDITIONS))
    s.add_argument("--no-oracle", action="store_true", help="skip translating the human transcript")
    s.add_argument("--bootstrap", type=int, default=1000)
    add_profile_args(s)
    s.set_defaults(func=cmd_speech)

    lt = ev.add_parser("latency", help="stream real speech at real-time pace; per-stage p50/p95/p99")
    lt.add_argument("--dataset", choices=["fleurs", "synthetic"], default="fleurs")
    lt.add_argument("--n", type=int, default=30)
    lt.add_argument("--offset", type=int, default=0, help="skip the first K recordings")
    lt.add_argument("--gap", type=float, default=2.0, help="seconds of silence between recordings")
    lt.add_argument("--no-tts", action="store_true")
    add_profile_args(lt)
    lt.set_defaults(func=cmd_latency)

    m = ev.add_parser("mt", help="text translation on FLORES-200 (human references)")
    m.add_argument("--dataset", choices=["flores200"], default="flores200")
    m.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    m.add_argument("--n", type=int, default=0, help="first N sentences (0 = all)")
    m.add_argument("--bootstrap", type=int, default=250)
    add_profile_args(m)
    m.set_defaults(func=cmd_mt)

    h = ev.add_parser("human-sheet", help="rating sheet for bilingual raters from a speech evaluation run")
    h.add_argument("--from", dest="run", required=True, help="speech evaluation run directory")
    h.add_argument("--condition", default="clean")
    h.add_argument("--n", type=int, default=60)
    h.add_argument("--seed", type=int, default=8)
    h.add_argument("--audio", action="store_true", help="also synthesize English audio for naturalness ratings")
    add_profile_args(h)
    h.set_defaults(func=cmd_human_sheet)

    hs = ev.add_parser("human-score", help="aggregate filled rating sheets (one CSV per rater)")
    hs.add_argument("sheets", nargs="+")
    hs.add_argument("--key", help="key.csv from human-sheet (adds adequacy by ASR quality)")
    hs.set_defaults(func=cmd_human_score)

    b = sub.add_parser("benchmark", help="quick suite: speech (FLEURS), latency and MT on small subsets")
    b.add_argument("--n-speech", type=int, default=50)
    b.add_argument("--n-latency", type=int, default=10)
    b.add_argument("--n-mt", type=int, default=100)
    add_profile_args(b)
    b.set_defaults(func=cmd_benchmark)


# -- helpers ---------------------------------------------------------------------------------------
def _setup(args: argparse.Namespace, extra: list[str] | None = None):
    from ..platform.resolve import resolve_runtime
    from ..platform.storage import prepare_storage

    cfg, runtime, fb = resolve_runtime(load_config(args.profile, list(args.overrides) + (extra or [])))
    paths = prepare_storage(cfg)
    return cfg, runtime, fb, Path(paths["output_dir"])


def _components(cfg, fb, *, tts: bool):
    from ..asr import create_asr
    from ..mt import create_translator
    from ..pipeline import Components
    from ..platform.storage import prepare_storage
    from ..vad import create_vad

    paths = prepare_storage(cfg)
    t0 = time.perf_counter()
    vad = create_vad(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_ms, str(paths["cache_dir"]), fb,
                     allow_fallback=cfg.fallback.policy != "fail")
    asr = create_asr(cfg, fb)
    tr = create_translator(cfg, fb)
    syn = None
    if tts:
        from ..tts import create_synthesizer
        syn = create_synthesizer(cfg, fb)
    info = {"vad": vad.name, "asr": asr.model_name, "mt": tr.model_name,
            "tts": syn.backend.name if syn else None, "load_s": round(time.perf_counter() - t0, 1)}
    print(f"models loaded in {info['load_s']} s: {info['asr']} | {info['mt']}" + (f" | {info['tts']}" if syn else ""),
          file=sys.stderr, flush=True)
    return Components(vad=vad, asr=asr, translator=tr, synthesizer=syn), info


def _meta(cfg, runtime, models: dict, dataset: dict) -> dict[str, Any]:
    from ..eval.metrics import NORMALIZATION

    return {"profile": cfg.profile, "device": runtime.device, "models": models, "asr_beam": cfg.asr.beam_size,
            "mt_beam": cfg.mt.beam_size, "vad": cfg.vad.backend, "corrections": cfg.asr.corrections_path,
            "dataset": dataset, "normalization": NORMALIZATION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _write(run_dir: Path, mode: str, cfg, runtime, fb, results: dict, report: str, rows: list[dict] | None,
           datasets: list[dict]) -> None:
    from ..utils.manifest import build_manifest, write_manifest

    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    if rows is not None:
        (run_dir / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
                                             encoding="utf-8")
    write_manifest(run_dir, build_manifest(run_id=run_dir.name, mode=mode,
                                           config=cfg.model_dump(mode="json", by_alias=True),
                                           runtime=runtime.to_dict(), fallbacks=fb, datasets=datasets))


def _progress(label: str):
    t0 = time.perf_counter()

    def cb(k: int, n: int, row: dict) -> None:
        if k == n or k % 10 == 0 or k == 1:
            el = time.perf_counter() - t0
            eta = el / k * (n - k)
            print(f"  {label}: {k}/{n}  {el:6.0f} s elapsed, ~{eta:5.0f} s left", file=sys.stderr, flush=True)
    return cb


def _speech_items(dataset: str, n: int, offset: int = 0):
    from ..eval.datasets import load_fleurs, load_synthetic

    if dataset == "fleurs":
        items, prov = load_fleurs(n=(n + offset) if n else 0)
    else:
        items, prov = load_synthetic(n=(n + offset) if n else 0)
    if offset:
        items = items[offset:]
        prov = {**prov, "items": len(items), "subset": f"recordings {offset + 1}-{offset + len(items)}"}
    return items, prov


# -- commands --------------------------------------------------------------------------------------
async def _speech(args, cfg, comps, items, conditions, bootstrap, oracle=True):
    from ..eval.speech import run_speech, score_speech

    out, all_rows = {}, []
    for cond in conditions:
        print(f"== condition {cond}: {len(items)} recordings", file=sys.stderr, flush=True)
        rows = await run_speech(cfg, items, comps, condition=cond, oracle=oracle, progress=_progress(cond))
        out[cond] = score_speech(rows, bootstrap=bootstrap)
        all_rows += rows
        a = out[cond]["asr"]
        s = out[cond].get("speech_to_english", {})
        print(f"   {cond}: WER {a['wer']:.3f}  CER {a['cer']:.3f}  chrF {s.get('chrf', '-')}  BLEU {s.get('bleu', '-')}",
              file=sys.stderr, flush=True)
    return out, all_rows


def cmd_speech(args: argparse.Namespace) -> int:
    from ..eval.perturb import CONDITIONS
    from ..eval.report import speech_report
    from ..utils.manifest import new_run_dir

    cfg, runtime, fb, out_dir = _setup(args)
    items, prov = _speech_items(args.dataset, args.n)
    comps, info = _components(cfg, fb, tts=False)
    conds, rows = asyncio.run(_speech(args, cfg, comps, items, args.conditions, args.bootstrap, not args.no_oracle))
    meta = _meta(cfg, runtime, info, prov)
    meta["conditions"] = {c: CONDITIONS[c][0] for c in args.conditions}
    results = {"meta": meta, "conditions": conds}
    run_dir = new_run_dir(out_dir, "eval-speech")
    report = speech_report(results)
    _write(run_dir, "eval-speech", cfg, runtime, fb, results, report, rows, [prov])
    print(report)
    print(f"-> {run_dir}")
    return 0


def cmd_latency(args: argparse.Namespace) -> int:
    from ..eval.report import latency_report
    from ..eval.speech import run_latency
    from ..utils.manifest import new_run_dir

    cfg, runtime, fb, out_dir = _setup(args)
    items, prov = _speech_items(args.dataset, args.n, args.offset)
    comps, info = _components(cfg, fb, tts=not args.no_tts)
    print(f"streaming {len(items)} recordings at real-time pace ...", file=sys.stderr, flush=True)
    r = asyncio.run(run_latency(cfg, items, comps, gap_s=args.gap, tts=not args.no_tts,
                                progress=_progress("units")))
    meta = _meta(cfg, runtime, info, prov)
    run_dir = new_run_dir(out_dir, "eval-latency")
    report = latency_report(r, meta)
    _write(run_dir, "eval-latency", cfg, runtime, fb, {"meta": meta, "latency": r}, report, None, [prov])
    print(report)
    print(f"-> {run_dir}")
    return 0


def cmd_mt(args: argparse.Namespace) -> int:
    from ..eval.datasets import load_flores
    from ..eval.report import mt_report
    from ..eval.speech import run_mt
    from ..mt import create_translator
    from ..utils.manifest import new_run_dir

    cfg, runtime, fb, out_dir = _setup(args)
    src, ref, prov = load_flores(split=args.split, n=args.n)
    tr = create_translator(cfg, fb)
    tr.warmup()
    print(f"translating {len(src)} FLORES-200 {args.split} sentences with {tr.model_name} ...", file=sys.stderr, flush=True)
    r = run_mt(tr, src, ref, bootstrap=args.bootstrap, progress=_progress("sentences"))
    hyps = r.pop("hypotheses")
    meta = _meta(cfg, runtime, {"mt": tr.model_name}, prov)
    run_dir = new_run_dir(out_dir, "eval-mt")
    report = mt_report(r, meta)
    rows = [{"src": s, "ref": e, "hyp": h} for s, e, h in zip(src, ref, hyps, strict=True)]
    _write(run_dir, "eval-mt", cfg, runtime, fb, {"meta": meta, "mt": r}, report, rows, [prov])
    print(report)
    print(f"-> {run_dir}")
    return 0


def cmd_human_sheet(args: argparse.Namespace) -> int:
    from ..eval.human import make_sheet

    run = Path(args.run)
    rows = [json.loads(line) for line in (run / "items.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r.get("condition") == args.condition]
    if not rows:
        print(f"no items for condition {args.condition!r} in {run}", file=sys.stderr)
        return 2
    synth, sr = None, 22050
    if args.audio:
        from ..tts import create_synthesizer
        from ..utils.manifest import FallbackLog

        cfg = load_config(args.profile, args.overrides)
        syn = create_synthesizer(cfg, FallbackLog())
        sr = cfg.tts.sample_rate

        def synth(text: str):
            return syn.synthesize(text).audio
    info = make_sheet(rows, run / "human_eval", n=args.n, seed=args.seed, synthesize=synth, sample_rate=sr)
    print(json.dumps(info, indent=2))
    print("Give raters rating_sheet.csv (and the audio folder) with docs/human_evaluation.md; keep key.csv.")
    return 0


def cmd_human_score(args: argparse.Namespace) -> int:
    from ..eval.human import score_sheets

    print(json.dumps(score_sheets(args.sheets, args.key), indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from ..eval.datasets import load_flores
    from ..eval.report import latency_report, mt_report, speech_report
    from ..eval.speech import run_latency, run_mt
    from ..utils.manifest import new_run_dir

    cfg, runtime, fb, out_dir = _setup(args)
    items, prov = _speech_items("fleurs", args.n_speech + args.n_latency)
    comps, info = _components(cfg, fb, tts=True)
    speech_items, latency_items = items[:args.n_speech], items[args.n_speech:]

    async def go():
        conds, rows = await _speech(args, cfg, comps, speech_items, ["clean"], 500)
        lat = await run_latency(cfg, latency_items, comps, progress=_progress("latency units"))
        return conds, rows, lat

    conds, rows, lat = asyncio.run(go())
    src, ref, mt_prov = load_flores(n=args.n_mt)
    mt = run_mt(comps.translator, src, ref, bootstrap=100, progress=_progress("mt"))
    mt.pop("hypotheses")
    meta = _meta(cfg, runtime, info, {**prov, "items": len(speech_items), "subset": f"first {len(speech_items)}"})
    run_dir = new_run_dir(out_dir, "benchmark")
    lat_meta = {**meta, "dataset": {**prov, "subset": f"recordings {args.n_speech + 1}-{len(items)}"}}
    report = "\n".join([speech_report({"meta": meta, "conditions": conds}), latency_report(lat, lat_meta),
                        mt_report(mt, {**meta, "dataset": mt_prov})])
    _write(run_dir, "benchmark", cfg, runtime, fb, {"meta": meta, "speech": conds, "latency": lat, "mt": mt},
           report, rows, [prov, mt_prov])
    print(report)
    print(f"-> {run_dir}")
    return 0
