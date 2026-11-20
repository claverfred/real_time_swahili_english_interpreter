"""Command-line entry point: ``python -m src.cli <command>``."""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from ..config import ConfigError, available_profiles, load_config
from ..platform.diagnose import run_diagnosis
from ..platform.resolve import FallbackRefused, resolve_runtime

_ICON = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "info": "[info]"}


def _add_profile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default=None, help=f"one of: {', '.join(available_profiles())}")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set asr.beam_size=3 (repeatable)")


def cmd_diagnose(args: argparse.Namespace) -> int:
    report = run_diagnosis()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        cpu = report["cpu"]
        print(f"Platform: {report['platform']}   OS: {report['os']}")
        print(f"CPU: {cpu['physical_cores']} cores / {cpu['logical_cores']} threads, "
              f"RAM {cpu['ram_total_gb']} GB")
        print()
        for c in report["checks"]:
            line = f"{_ICON[c['status']]} {c['name']:<26} {c['detail']}"
            print(line)
            if c["hint"] and c["status"] in ("warn", "fail"):
                print(f"{'':>34}-> {c['hint']}")
        s = report["summary"]
        print(f"\n{s['ok']} ok, {s['warn']} warnings, {s['fail']} failures")
    return 1 if report["summary"]["fail"] else 0


def cmd_config(args: argparse.Namespace) -> int:
    if args.action == "list":
        for name in available_profiles():
            print(name)
        return 0
    cfg = load_config(args.profile, args.overrides)
    if args.action == "validate":
        print(f"profile {cfg.profile!r} is valid")
        return 0
    payload: dict[str, object] = {"config": cfg.model_dump(mode="json", by_alias=True)}
    if args.resolve:
        cfg, runtime, fallbacks = resolve_runtime(cfg)
        payload = {
            "config": cfg.model_dump(mode="json", by_alias=True),
            "runtime": runtime.to_dict(),
            "fallbacks": fallbacks.to_list(),
            "cloud_services": cfg.uses_cloud(),
        }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    from ..capture import list_devices

    devices = list_devices()
    if args.json:
        print(json.dumps(devices, indent=2))
        return 0
    print(f"{'#':>3}  {'in':>2} {'out':>3}  {'rate':>6}  {'host API':<22} name")
    for d in devices:
        mark = ("*in " if d["default_input"] else "") + ("*out" if d["default_output"] else "")
        print(f"{d['index']:>3}  {d['max_input_channels']:>2} {d['max_output_channels']:>3}  "
              f"{int(d['default_samplerate']):>6}  {d['hostapi'][:22]:<22} {d['name']} {mark}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from ..audio import analyze_levels, load_audio

    cfg = load_config(args.profile, args.overrides)
    audio, info = load_audio(args.input, cfg.audio.sample_rate, cfg.audio.max_input_seconds,
                             cfg.audio.max_input_mb)
    print(json.dumps({"file": info.to_dict(), "levels": analyze_levels(audio, cfg.audio.sample_rate)},
                     indent=2))
    return 0


def cmd_vad(args: argparse.Namespace) -> int:
    import time
    from pathlib import Path

    from ..audio import analyze_levels, load_audio, save_wav
    from ..pipeline.events import EventRecorder
    from ..platform.storage import prepare_storage
    from ..utils.manifest import build_manifest, new_run_dir, write_manifest
    from ..vad import VadSegmenter, create_vad, segment_audio

    overrides = list(args.overrides) + ([f"vad.backend={args.backend}"] if args.backend else [])
    cfg, runtime, fb = resolve_runtime(load_config(args.profile, overrides))
    paths = prepare_storage(cfg)
    sr = cfg.audio.sample_rate
    audio, info = load_audio(args.input, sr, cfg.audio.max_input_seconds, cfg.audio.max_input_mb, fb)
    levels = analyze_levels(audio, sr)

    t_load = time.perf_counter()
    backend = create_vad(cfg.vad, sr, cfg.audio.frame_ms, str(paths["cache_dir"]), fb,
                         allow_fallback=cfg.fallback.policy != "fail")
    load_ms = (time.perf_counter() - t_load) * 1000
    recorder = EventRecorder(log_text=cfg.logging.log_text)
    t0 = time.perf_counter()
    segments = segment_audio(audio, VadSegmenter(backend, cfg.vad, recorder))
    proc_s = time.perf_counter() - t0

    speech_s = sum(s.duration_s for s in segments)
    timing = {"model_load_ms": round(load_ms, 1), "processing_s": round(proc_s, 4),
              "rtf": round(proc_s / max(info.duration_s, 1e-9), 5)}
    for w in levels["warnings"]:  # type: ignore[union-attr]
        print(f"warning: {w}", file=sys.stderr)
    for rec in fb.records:
        print(f"fallback: {rec.stage} {rec.requested} -> {rec.selected} ({rec.reason})", file=sys.stderr)
    print(f"VAD backend: {backend.name}   file: {Path(args.input).name} ({info.duration_s:.2f} s)")
    print(f"{len(segments)} segments, {speech_s:.2f} s speech, "
          f"load {timing['model_load_ms']} ms, RTF {timing['rtf']}")
    for s in segments:
        print(f"  #{s.segment_id:<3} {s.start_s:8.3f} -> {s.end_s:8.3f}  ({s.duration_s:6.3f} s)  "
              f"{s.reason:<12} p={s.mean_prob:.2f}")

    if args.export:
        run_dir = new_run_dir(paths["output_dir"], "vad")
        (run_dir / "segments.json").write_text(json.dumps([s.to_dict() for s in segments], indent=2),
                                               encoding="utf-8")
        recorder.export_jsonl(run_dir / "events.jsonl")
        recorder.export_csv(run_dir / "events.csv")
        if args.save_segments:
            for s in segments:
                save_wav(run_dir / "segments" / f"seg_{s.segment_id:04d}.wav", s.audio, sr)
        manifest = build_manifest(
            run_id=run_dir.name, mode="vad", config=cfg.model_dump(mode="json", by_alias=True),
            runtime=runtime.to_dict(), fallbacks=fb,
            extra={"input": info.to_dict(), "levels": levels, "timing": timing, "vad_backend": backend.name},
        )
        write_manifest(run_dir, manifest)
        print(f"exported to {run_dir}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    import asyncio

    import numpy as np

    from ..audio import analyze_levels, save_wav
    from ..capture import CONSENT_NOTICE, MicrophoneStream
    from ..utils.manifest import FallbackLog

    print(CONSENT_NOTICE, file=sys.stderr)
    sr, fb = 16000, FallbackLog()

    chunks: list[np.ndarray] = []
    mic = MicrophoneStream(args.device, sr, fallbacks=fb)

    async def run() -> None:
        got = 0
        async with mic:
            print(f"Recording {args.seconds} s ... (Ctrl+C to stop early and keep the audio)",
                  file=sys.stderr)
            async for chunk in mic:
                chunks.append(chunk)
                got += len(chunk)
                if got >= args.seconds * sr:
                    break

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("stopped early", file=sys.stderr)
    if not chunks:
        print("error: no audio was captured (check the device index and Windows microphone permissions)",
              file=sys.stderr)
        return 3
    audio = np.concatenate(chunks)[: int(args.seconds * sr)]
    path = save_wav(args.output, audio, sr)
    print(json.dumps({"output": str(path), "levels": analyze_levels(audio, sr),
                      "capture": vars(mic.stats), "fallbacks": fb.to_list()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli",
                                     description="Real-time Swahili-English interpreter")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("diagnose", help="check Python, packages, audio, CUDA, RAM, disk")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("config", help="list, show, or validate configuration profiles")
    p.add_argument("action", choices=["list", "show", "validate"])
    p.add_argument("--resolve", action="store_true", help="resolve auto device/precision for this machine")
    p.add_argument("--json", action="store_true")
    _add_profile_args(p)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("devices", help="list audio input/output devices")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("inspect", help="validate an audio file: format, duration, levels, clipping")
    p.add_argument("--input", required=True)
    _add_profile_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("vad", help="segment an audio file into speech utterances")
    p.add_argument("--input", required=True)
    p.add_argument("--backend", choices=["silero", "webrtc", "energy"], help="shortcut for --set vad.backend=")
    p.add_argument("--export", action="store_true", help="write segments, events and manifest to a new run dir")
    p.add_argument("--save-segments", action="store_true", help="with --export: also save each segment as WAV")
    _add_profile_args(p)
    p.set_defaults(func=cmd_vad)

    p = sub.add_parser("record", help="record a test clip from the microphone (local only)")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", type=int, default=None, help="input device index (see 'devices')")
    p.set_defaults(func=cmd_record)

    from .transcribe import add_parser as add_transcribe

    add_transcribe(sub, _add_profile_args)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, FallbackRefused) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Known runtime errors (bad audio, missing device/backend) -> clean message, exit 3.
        from ..asr import AsrError
        from ..audio import AudioError
        from ..capture import CaptureError
        from ..vad import VadUnavailable

        if isinstance(exc, (AudioError, CaptureError, VadUnavailable, AsrError)):
            print(f"error: {exc}", file=sys.stderr)
            return 3
        raise
