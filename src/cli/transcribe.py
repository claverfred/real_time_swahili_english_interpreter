"""``transcribe``: ASR-only mode (batch file, simulated live stream, or live microphone)."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import numpy as np

from ..asr import AsrUpdate, StreamingTranscriber, create_asr
from ..audio import analyze_levels, load_audio
from ..config import load_config
from ..pipeline.events import EventRecorder
from ..platform.resolve import resolve_runtime
from ..platform.storage import prepare_storage
from ..utils.manifest import build_manifest, new_run_dir, write_manifest
from ..utils.stats import summarize
from ..vad import VadSegmenter, create_vad


def add_parser(sub: argparse._SubParsersAction, add_profile_args) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("transcribe", help="Swahili speech -> Swahili text (ASR-only mode)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="audio file")
    src.add_argument("--mic", action="store_true", help="live microphone")
    p.add_argument("--seconds", type=float, default=30.0, help="with --mic: how long to listen")
    p.add_argument("--device", type=int, default=None, help="with --mic: input device index")
    p.add_argument("--live-sim", action="store_true", help="stream the file at real-time pace with partials")
    p.add_argument("--partials", action="store_true", help="emit partial hypotheses in batch mode too")
    p.add_argument("--no-vad", action="store_true", help="decode the whole file in one call (no streaming)")
    p.add_argument("--model", help="shortcut for --set asr.model=")
    p.add_argument("--reference", help="reference transcript text, or @file.txt, to compute WER/CER")
    p.add_argument("--export", action="store_true", help="write transcript, events and manifest to a run dir")
    p.add_argument("--quiet", action="store_true", help="do not print partials")
    add_profile_args(p)
    p.set_defaults(func=cmd_transcribe)


def _print_update(u: AsrUpdate, quiet: bool) -> None:
    if u.kind == "partial":
        if not quiet:
            print(f"  … [{u.segment_id}] {u.stable} \x1b[2m{u.unstable}\x1b[0m  ({u.latency_ms:.0f} ms)",
                  file=sys.stderr)
        return
    flags = []
    if u.extra.get("uncertain"):
        flags.append("UNCERTAIN")
    if u.extra.get("rejected"):
        flags.append(f"rejected={u.extra['rejected']}")
    lag = f", final lag {u.extra['final_lag_ms']:.0f} ms" if "final_lag_ms" in u.extra else ""
    print(f"[{u.segment_id:>3}] {u.start_s:7.2f}-{u.end_s:7.2f}  {u.text or '∅'}"
          f"   ({u.latency_ms:.0f} ms{lag}){' ' + ' '.join(flags) if flags else ''}")


def cmd_transcribe(args: argparse.Namespace) -> int:
    overrides = list(args.overrides) + ([f"asr.model={args.model}"] if args.model else [])
    cfg, runtime, fb = resolve_runtime(load_config(args.profile, overrides))
    paths = prepare_storage(cfg)
    sr = cfg.audio.sample_rate

    print(f"Loading ASR {cfg.asr.model} on {runtime.device}/{cfg.asr.compute_type} "
          f"({runtime.cpu_threads} threads) ...", file=sys.stderr)
    asr = create_asr(cfg, fb)
    cfg.asr.model = asr.model_name
    timing: dict[str, object] = dict(asr.load_info)
    timing["first_inference_ms"] = round(asr.warmup(), 1)       # cold decode
    timing["second_inference_ms"] = round(asr.warmup(), 1)      # warm decode
    for rec in fb.records:
        print(f"fallback: {rec.stage} {rec.requested} -> {rec.selected} ({rec.reason})", file=sys.stderr)

    recorder = EventRecorder(log_text=cfg.logging.log_text)
    updates: list[AsrUpdate] = []
    info = None
    levels = None

    if args.input:
        audio, info = load_audio(args.input, sr, cfg.audio.max_input_seconds, cfg.audio.max_input_mb, fb)
        levels = analyze_levels(audio, sr)
        for w in levels["warnings"]:  # type: ignore[union-attr]
            print(f"warning: {w}", file=sys.stderr)

    t0 = time.perf_counter()
    if args.no_vad:
        assert info is not None, "--no-vad needs --input"
        res = asr.transcribe(audio)
        updates.append(AsrUpdate("final", 0, res.text, res.text, latency_ms=res.inference_ms,
                                 audio_s=res.duration_s, result=res, end_s=res.duration_s,
                                 extra={"rejected": [s.rejected for s in res.rejected_segments],
                                        "uncertain": res.uncertain}))
        _print_update(updates[-1], args.quiet)
    else:
        vad = create_vad(cfg.vad, sr, cfg.audio.frame_ms, str(paths["cache_dir"]), fb,
                         allow_fallback=cfg.fallback.policy != "fail")
        segmenter = VadSegmenter(vad, cfg.vad, recorder)
        live = args.mic or args.live_sim
        streamer = StreamingTranscriber(asr, cfg.asr, sr, recorder, partials=live or args.partials)

        def handle(chunk: np.ndarray | None) -> list[AsrUpdate]:
            evs = segmenter.flush() if chunk is None else segmenter.process(chunk)
            out = []
            for ev in evs:
                for u in streamer.on_vad_event(ev):
                    if u.kind == "final" and live:
                        # end of speech -> final text available, as experienced by a listener
                        u.extra["final_lag_ms"] = round((time.perf_counter() - t0 - u.end_s) * 1000, 1)
                    _print_update(u, args.quiet)
                    out.append(u)
            return out

        if live:
            updates = asyncio.run(_run_live(args, cfg, sr, audio if args.input else None, handle, fb))
        else:
            step = int(0.5 * sr)
            for i in range(0, len(audio), step):
                updates += handle(audio[i:i + step])
            updates += handle(None)
    wall_s = time.perf_counter() - t0

    finals = [u for u in updates if u.kind == "final"]
    partials = [u for u in updates if u.kind == "partial"]
    duration = info.duration_s if info else float(args.seconds)
    transcript = " ".join(u.text for u in finals if u.text).strip()
    final_lat = [u.latency_ms for u in finals if u.latency_ms > 0]
    summary: dict[str, object] = {
        "model": asr.model_name, "device": runtime.device, "compute_type": cfg.asr.compute_type,
        "audio_s": round(duration, 3), "wall_s": round(wall_s, 3),
        "rtf_total": round(sum(final_lat) / 1000 / duration, 4) if duration else None,
        "segments": len(finals), "empty_segments": sum(1 for u in finals if not u.text),
        "uncertain_segments": sum(1 for u in finals if u.extra.get("uncertain")),
        "partials": len(partials), "revisions": sum(u.revisions for u in finals),
        "final_decode_ms": summarize(final_lat),
        "partial_decode_ms": summarize([u.latency_ms for u in partials]),
        "timing": timing,
    }
    lags = [float(u.extra["final_lag_ms"]) for u in finals if "final_lag_ms" in u.extra]  # type: ignore[arg-type]
    if lags:
        summary["final_lag_ms"] = summarize(lags)
    if args.reference:
        from ..eval.metrics import wer_cer

        ref = Path(args.reference[1:]).read_text(encoding="utf-8") if args.reference.startswith("@") else args.reference
        summary["accuracy"] = wer_cer([ref], [transcript])

    print("\nTranscript:", transcript or "∅")
    print(json.dumps(summary, indent=2, default=str), file=sys.stderr)

    if args.export:
        run_dir = new_run_dir(paths["output_dir"], "asr")
        (run_dir / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")
        (run_dir / "transcript.json").write_text(json.dumps(
            [{"segment_id": u.segment_id, "start_s": u.start_s, "end_s": u.end_s, "text": u.text,
              "latency_ms": u.latency_ms, "revisions": u.revisions, **u.extra,
              "asr": u.result.to_dict() if u.result else None} for u in finals],
            indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        _write_csv(run_dir / "segments.csv", finals)
        recorder.export_jsonl(run_dir / "events.jsonl")
        recorder.export_csv(run_dir / "events.csv")
        write_manifest(run_dir, build_manifest(
            run_id=run_dir.name, mode="asr-live" if (args.mic or args.live_sim) else "asr-batch",
            config=cfg.model_dump(mode="json", by_alias=True), runtime=runtime.to_dict(), fallbacks=fb,
            extra={"input": info.to_dict() if info else {"source": "microphone"}, "levels": levels,
                   "summary": summary}))
        print(f"exported to {run_dir}", file=sys.stderr)
    return 0


def _write_csv(path: Path, finals: Iterable[AsrUpdate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "start_s", "end_s", "text", "latency_ms", "uncertain", "rejected", "revisions"])
        for u in finals:
            w.writerow([u.segment_id, round(u.start_s, 3), round(u.end_s, 3), u.text, u.latency_ms,
                        u.extra.get("uncertain"), ";".join(map(str, u.extra.get("rejected", []))), u.revisions])


async def _run_live(args, cfg, sr, audio, handle, fb) -> list[AsrUpdate]:
    """Feed chunks from the mic or a paced file; decode in a worker thread so capture never stalls."""
    from ..capture import CONSENT_NOTICE, MicrophoneStream, stream_array

    loop = asyncio.get_running_loop()
    updates: list[AsrUpdate] = []

    async def source() -> AsyncIterator[np.ndarray]:
        if audio is not None:
            async for c in stream_array(audio, sr, block_ms=100, realtime_factor=1.0):
                yield c
            return
        print(CONSENT_NOTICE, file=sys.stderr)
        got = 0
        async with MicrophoneStream(args.device, sr, fallbacks=fb) as mic:
            print(f"Listening for {args.seconds:.0f} s — speak Swahili (Ctrl+C to stop)", file=sys.stderr)
            async for c in mic:
                yield c
                got += len(c)
                if got >= args.seconds * sr:
                    break

    # Decoding runs in one worker thread; chunks queue up while it is busy (bounded by queue_max).
    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(cfg.streaming.queue_max)

    async def consume() -> None:
        while True:
            chunk = await queue.get()
            updates.extend(await loop.run_in_executor(None, handle, chunk))
            if chunk is None:
                return

    worker = asyncio.create_task(consume())
    try:
        async for c in source():
            await queue.put(c)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("stopped", file=sys.stderr)
    await queue.put(None)
    await worker
    return updates
