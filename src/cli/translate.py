"""``translate``: Swahili text -> English text (MT-only mode).

    python -m src.cli translate --text "Habari za asubuhi." --profile cpu
    python -m src.cli translate --input sentences.txt --context --export

Text is stabilized and split into sentences exactly as the live pipeline does. With
``--context`` the previous sentences are used as translation context (mt.context_mode=prefix).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import load_config
from ..mt import MtStage, create_translator
from ..pipeline.events import EventRecorder
from ..platform.resolve import resolve_runtime
from ..platform.storage import prepare_storage
from ..text import SentenceAggregator, stabilize
from ..utils.manifest import build_manifest, new_run_dir, write_manifest
from ..utils.stats import summarize


def add_parser(sub: argparse._SubParsersAction, add_profile_args) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("translate", help="Swahili text -> English text (MT-only mode)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", action="append", help="Swahili text (repeatable)")
    src.add_argument("--input", help="UTF-8 text file; each non-empty line is one utterance")
    p.add_argument("--model", help="shortcut for --set mt.model=")
    p.add_argument("--engine", choices=["auto", "ctranslate2", "transformers"], help="shortcut for mt.engine")
    p.add_argument("--beam", type=int, help="shortcut for mt.beam_size")
    p.add_argument("--context", action="store_true", help="use previous sentences as context")
    p.add_argument("--glossary", help="glossary file (csv/json/yaml); shortcut for mt.glossary_path")
    p.add_argument("--mask", action="store_true", help="mask numbers/glossary terms (mt.entity_strategy=mask)")
    p.add_argument("--export", action="store_true", help="write translations, events and manifest")
    p.add_argument("--json", action="store_true", help="print results as JSON lines")
    add_profile_args(p)
    p.set_defaults(func=cmd_translate)


def _overrides(args: argparse.Namespace) -> list[str]:
    o = list(args.overrides)
    if args.model:
        o.append(f"mt.model={args.model}")
    if args.engine:
        o.append(f"mt.engine={args.engine}")
    if args.beam:
        o.append(f"mt.beam_size={args.beam}")
    if args.context:
        o.append("mt.context_mode=prefix")
    if args.glossary:
        o.append(f"mt.glossary_path={args.glossary}")
    if args.mask:
        o.append("mt.entity_strategy=mask")
    return o


def cmd_translate(args: argparse.Namespace) -> int:
    cfg, runtime, fb = resolve_runtime(load_config(args.profile, _overrides(args)))
    paths = prepare_storage(cfg)
    lines = args.text or [ln for ln in Path(args.input).read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

    print(f"Loading MT {cfg.mt.model} ({cfg.mt.engine}) on {runtime.device}/{cfg.mt.compute_type} ...",
          file=sys.stderr)
    translator = create_translator(cfg, fb)
    cfg.mt.model = translator.model_name
    timing = dict(translator.load_info)
    timing["first_inference_ms"] = round(translator.warmup(), 1)
    timing["second_inference_ms"] = round(translator.warmup(), 1)
    for rec in fb.records:
        print(f"fallback: {rec.stage} {rec.requested} -> {rec.selected} ({rec.reason})", file=sys.stderr)

    recorder = EventRecorder(log_text=cfg.logging.log_text)
    stage = MtStage(translator, cfg.mt, cfg.text, recorder)
    agg = SentenceAggregator(cfg.text)
    rows = []
    for i, line in enumerate(lines):
        st = stabilize(line, cfg.text, final=True)
        for unit in agg.push(st.text, segment_id=i, start_s=0.0, end_s=0.0, boundary=True):
            up = stage._translate_unit(unit)  # same path as the live pipeline, without ASR
            res = up.result
            row = {"line": i, "source": unit.text, "english": res.text, "latency_ms": res.latency_ms,
                   "context_used": res.context_used, "cached": res.cached, "flags": res.flags,
                   "preservation": res.preservation}
            rows.append(row)
            if args.json:
                print(json.dumps(row, ensure_ascii=False))
            else:
                flags = f"   [{', '.join(res.flags)}]" if res.flags else ""
                print(f"SW: {unit.text}\nEN: {res.text}   ({res.latency_ms:.0f} ms){flags}\n")

    summary = {"model": translator.model_name, "engine": timing.get("engine"), "units": len(rows),
               "latency_ms": summarize([r["latency_ms"] for r in rows if not r["cached"]]),
               "cache": {"hits": translator.cache_hits, "misses": translator.cache_misses},
               "flag_counts": {f: sum(f in r["flags"] for r in rows) for f in sorted({f for r in rows for f in r["flags"]})},
               "timing": timing}
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)

    if args.export:
        run_dir = new_run_dir(paths["output_dir"], "mt")
        (run_dir / "translations.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (run_dir / "translation.txt").write_text("\n".join(r["english"] for r in rows) + "\n", encoding="utf-8")
        recorder.export_jsonl(run_dir / "events.jsonl")
        write_manifest(run_dir, build_manifest(
            run_id=run_dir.name, mode="translate", config=cfg.model_dump(mode="json", by_alias=True),
            runtime=runtime.to_dict(), fallbacks=fb, extra={"summary": summary}))
        print(f"exported to {run_dir}", file=sys.stderr)
    return 0
