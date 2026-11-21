"""``speak``: English text -> English speech (TTS-only mode).

    python -m src.cli speak --text "Good morning. The price is 2,500 shillings." --output out.wav
    python -m src.cli speak --text "Hello" --backend sapi --play
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..audio import save_wav
from ..config import load_config
from ..pipeline.events import EventRecorder
from ..platform.resolve import resolve_runtime
from ..platform.storage import prepare_storage
from ..tts import TtsStage, create_synthesizer
from ..utils.manifest import build_manifest, new_run_dir, write_manifest
from ..utils.stats import summarize


def add_parser(sub: argparse._SubParsersAction, add_profile_args) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("speak", help="English text -> English speech (TTS-only mode)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", action="append", help="English text (repeatable)")
    src.add_argument("--input", help="UTF-8 text file; each non-empty line is one utterance")
    p.add_argument("--backend", help="shortcut for tts.backend (piper, kokoro, sapi, mms, edge)")
    p.add_argument("--voice", help="shortcut for tts.voice")
    p.add_argument("--output", help="write the concatenated audio to this WAV file")
    p.add_argument("--play", action="store_true", help="play through the default output device")
    p.add_argument("--export", action="store_true", help="write audio, events and manifest to a run dir")
    add_profile_args(p)
    p.set_defaults(func=cmd_speak)


def cmd_speak(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.backend:
        overrides += [f"tts.backend={args.backend}", "tts.fallback_backends=[]"]
    if args.voice:
        overrides.append(f"tts.voice={args.voice}")
    elif args.backend:
        base = load_config(args.profile, list(args.overrides))
        if args.backend != base.tts.backend:
            voice = base.tts.voices.get(args.backend)
            if voice is None:
                print(f"error: no default voice for {args.backend}; pass --voice", file=sys.stderr)
                return 2
            overrides.append(f"tts.voice={voice}")
    cfg, runtime, fb = resolve_runtime(load_config(args.profile, overrides))
    paths = prepare_storage(cfg)
    lines = args.text or [ln for ln in Path(args.input).read_text(encoding="utf-8-sig").splitlines() if ln.strip()]

    print(f"Loading TTS {cfg.tts.backend} ({cfg.tts.voice}) ...", file=sys.stderr)
    synth = create_synthesizer(cfg, fb)
    timing = dict(synth.load_info)
    timing["first_inference_ms"] = round(synth.warmup(), 1)
    timing["second_inference_ms"] = round(synth.warmup(), 1)
    if synth.backend.is_cloud:
        print("note: cloud TTS — the English text is sent to the provider", file=sys.stderr)
    for rec in fb.records:
        print(f"fallback: {rec.stage} {rec.requested} -> {rec.selected} ({rec.reason})", file=sys.stderr)

    recorder = EventRecorder(log_text=cfg.logging.log_text)
    stage = TtsStage(synth, recorder)
    pieces, rows = [], []
    for i, line in enumerate(lines):
        for up in stage.speak(line, unit_id=i):
            c = up.chunk
            pieces += [c.audio, synth.pause()]
            rows.append({"unit": i, "chunk": c.index, "text": c.text, "spoken": c.spoken, "backend": c.backend,
                         "synth_ms": c.synth_ms, "audio_s": round(c.duration_s, 3), "rtf": round(c.rtf, 4),
                         "first_audio_ms": up.first_audio_ms if up.is_first else None, "warnings": c.warnings})
            warn = f"  [{', '.join(c.warnings)}]" if c.warnings else ""
            print(f"[{i}.{c.index}] {c.synth_ms:6.0f} ms  {c.duration_s:5.2f} s  rtf {c.rtf:.3f}  {c.text}{warn}")

    import numpy as np
    audio = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32)
    firsts = [r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None]
    summary = {"backend": synth.backend.name, "voice": cfg.tts.voice, "license": synth.backend.license,
               "sample_rate": synth.sample_rate, "chunks": len(rows), "audio_s": round(len(audio) / synth.sample_rate, 3),
               "first_audio_ms": summarize(firsts), "chunk_synth_ms": summarize([r["synth_ms"] for r in rows]),
               "rtf": round(sum(r["synth_ms"] for r in rows) / 1000 / max(sum(r["audio_s"] for r in rows), 1e-9), 4),
               "warnings": sum(bool(r["warnings"]) for r in rows), "switches": synth.switches, "timing": timing}
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)

    if args.output:
        print(f"wrote {save_wav(args.output, audio, synth.sample_rate)}", file=sys.stderr)
    if args.export:
        run_dir = new_run_dir(paths["output_dir"], "tts")
        save_wav(run_dir / "speech.wav", audio, synth.sample_rate)
        (run_dir / "chunks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")
        recorder.export_jsonl(run_dir / "events.jsonl")
        write_manifest(run_dir, build_manifest(
            run_id=run_dir.name, mode="speak", config=cfg.model_dump(mode="json", by_alias=True),
            runtime=runtime.to_dict(), fallbacks=fb, extra={"summary": summary}))
        print(f"exported to {run_dir}", file=sys.stderr)
    if args.play:
        from ..audio.playback import play
        play(audio, synth.sample_rate)
    return 0
