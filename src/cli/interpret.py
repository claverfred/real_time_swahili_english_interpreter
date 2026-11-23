"""``interpret``: Swahili speech -> English text and English speech (full pipeline).

    python -m src.cli interpret --input audio.wav --profile cpu                # batch, as fast as possible
    python -m src.cli interpret --input audio.wav --live-sim --play            # real-time pace, speaker out
    python -m src.cli interpret --mic --seconds 60 --play                      # live microphone
    python -m src.cli interpret --input audio.wav --no-tts --export            # text only, export run dir

Latencies are measured from the end of each utterance. They describe what a listener
experiences only with real-time pacing (``--live-sim`` or ``--mic``); in batch mode the audio
arrives faster than real time, so they include the backlog.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import numpy as np

from ..config import load_config
from ..pipeline import EventType, RealtimeSwahiliInterpreter
from ..platform.storage import prepare_storage
from ..utils.manifest import build_manifest, new_run_dir, write_manifest

DIM, RESET = "\x1b[2m", "\x1b[0m"


def add_parser(sub: argparse._SubParsersAction, add_profile_args) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("interpret", help="Swahili speech -> English text + speech (full pipeline)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="audio file")
    src.add_argument("--mic", action="store_true", help="live microphone")
    p.add_argument("--seconds", type=float, default=60.0, help="with --mic: how long to listen")
    p.add_argument("--device", type=int, default=None, help="with --mic: input device index")
    p.add_argument("--output-device", type=int, default=None, help="with --play: output device index")
    p.add_argument("--live-sim", action="store_true", help="feed the file at real-time pace")
    p.add_argument("--partials", action="store_true", help="partial hypotheses in batch mode too")
    p.add_argument("--no-tts", action="store_true", help="text only (skip speech synthesis)")
    p.add_argument("--play", action="store_true", help="play English speech on the local speaker")
    p.add_argument("--output", help="write the English speech to this WAV file")
    p.add_argument("--export", action="store_true", help="write transcripts, audio, events and manifest")
    p.add_argument("--quiet", action="store_true", help="do not print partials or per-unit latencies")
    p.add_argument("--mode", choices=["consecutive", "simultaneous"], default=None,
                   help="with --mic --play: consecutive (default) closes the mic until each English "
                        "translation has been spoken; simultaneous keeps listening (use headphones)")
    p.add_argument("--no-echo-guard", action="store_true",
                   help="with --mic --play: keep the mic open while English plays (only with headphones)")
    p.add_argument("--no-cue", action="store_true",
                   help="with --mic: no spoken 'speak now' prompt and beeps at start and end")
    add_profile_args(p)
    p.set_defaults(func=cmd_interpret)


def _print_event(ev: Any, quiet: bool) -> None:
    t, d = ev.type, ev.data
    if t == EventType.VAD_SPEECH_START:
        print("  (hearing speech ...)", file=sys.stderr)
    elif t == EventType.VAD_SPEECH_END:
        print("  (pause detected — interpreting ...)", file=sys.stderr)
    elif t == EventType.ASR_PARTIAL and not quiet:
        print(f"  {DIM}… sw[{ev.segment_id}] {ev.text}{RESET}", file=sys.stderr)
    elif t == EventType.ASR_FINAL:
        flag = "  UNCERTAIN" if d.get("uncertain") else ""
        print(f"SW [{ev.segment_id:>3}] {d.get('start_s', 0):7.2f}-{d.get('end_s', 0):7.2f}  {ev.text or '∅'}{flag}")
    elif t == EventType.MT_PARTIAL and not quiet:
        print(f"  {DIM}… en (provisional) {ev.text}{RESET}", file=sys.stderr)
    elif t == EventType.MT_FINAL:
        flags = f"   [{', '.join(d['flags'])}]" if d.get("flags") else ""
        print(f"EN [{d.get('unit_id'):>3}] {ev.text}{flags}")
    elif t == EventType.PIPELINE_COMPLETE and not quiet:
        parts = [f"{name} {d[k]:.0f} ms" for name, k in (("asr", "lat_asr_final_ms"), ("mt", "lat_mt_final_ms"),
                                                         ("first audio", "lat_first_audio_ms"))
                 if d.get(k) is not None]
        err = f"  TTS FAILED: {d['tts_error']}" if d.get("tts_error") else ""
        print(f"  {DIM}after speech end: {' · '.join(parts)}{RESET}{err}", file=sys.stderr)
    elif t == EventType.FALLBACK:
        print(f"fallback: {d.get('stage')} {d.get('requested', d.get('from'))} -> "
              f"{d.get('selected', d.get('to'))} ({d.get('reason')})", file=sys.stderr)
    elif t == EventType.ERROR:
        print(f"error [{d.get('stage')}]: {d.get('error')}", file=sys.stderr)
    elif t == EventType.STATUS and d.get("status") == "audio_overflow":
        print("warning: pipeline full, audio dropped", file=sys.stderr)


def _beep(sr: int, n: int = 1) -> np.ndarray:
    t = np.arange(int(0.15 * sr)) / sr
    tone = (0.25 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.float32)
    gap = np.zeros(int(0.1 * sr), np.float32)
    return np.concatenate([np.concatenate([tone, gap]) for _ in range(n)])


async def _play_cue(interp: Any, device: int | None, text: str | None, beeps: int = 1) -> None:
    """Spoken prompt (via the session's TTS voice) plus beep(s); failures never stop the session."""
    from ..audio.playback import play

    parts, sr = [], interp.cfg.tts.sample_rate
    if text:
        said = await interp.say(text)
        if said is not None:
            speech, sr = said
            parts += [speech, np.zeros(int(0.2 * sr), np.float32)]
    parts.append(_beep(sr, beeps))
    try:
        await asyncio.get_running_loop().run_in_executor(None, lambda: play(np.concatenate(parts), sr, device=device))
    except Exception as exc:
        print(f"warning: could not play the cue: {exc}", file=sys.stderr)


TURN_END_SILENCE_MS = 1200   # consecutive mode: a breath mid-sentence must not end the turn


def cmd_interpret(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    consecutive = bool(args.mic and args.play and not args.no_tts) and (args.mode or "consecutive") == "consecutive"
    if consecutive and not any(o.startswith("vad.min_silence_ms=") for o in overrides):
        # The mic closes when the turn ends, so ending it at the first short pause loses the rest
        # of the sentence (observed: utterances cut to 1.6-2.5 s).
        overrides.insert(0, f"vad.min_silence_ms={TURN_END_SILENCE_MS}")
    cfg = load_config(args.profile, overrides)
    return asyncio.run(_run(args, cfg))


async def _run(args: argparse.Namespace, cfg: Any) -> int:
    from ..audio import analyze_levels, load_audio, save_wav

    live = bool(args.mic or args.live_sim)
    # Partials follow asr.partials in live modes; batch mode skips them unless asked (pure overhead).
    partials = True if args.partials else (None if live else False)
    interp = RealtimeSwahiliInterpreter(cfg, enable_tts=not args.no_tts, partials=partials)
    cfg, sr = interp.cfg, interp.sample_rate
    paths = prepare_storage(cfg)
    audio, info, levels = None, None, None
    if args.input:
        audio, info = load_audio(args.input, sr, cfg.audio.max_input_seconds, cfg.audio.max_input_mb, interp.fallbacks)
        levels = analyze_levels(audio, sr)
        for w in levels["warnings"]:  # type: ignore[union-attr]
            print(f"warning: {w}", file=sys.stderr)

    print(f"Loading models (profile {cfg.profile}, {interp.runtime.device}) ...", file=sys.stderr)
    await interp.start()
    models = interp.model_status()
    tts = models.get("tts")
    print(f"ASR {models['asr']['model']} | MT {models['mt']['model']} | "
          f"TTS {tts['backend'] + ' (' + tts['voice'] + ')' if tts else 'off'}", file=sys.stderr)
    if tts and tts.get("cloud"):
        print("note: cloud TTS — the English text is sent to the provider", file=sys.stderr)

    player = None
    if args.play and not args.no_tts:
        from ..pipeline.playback import AudioPlayer
        player = AudioPlayer(interp.recorder, device=args.output_device, pause_ms=cfg.tts.sentence_pause_ms)
        player.start()

    speech: list[np.ndarray] = []
    out_sr = cfg.tts.sample_rate
    pause = np.zeros(int(out_sr * cfg.tts.sentence_pause_ms / 1000), np.float32)

    # Turn-taking (consecutive interpretation): after each utterance the mic stays closed until its
    # English has been spoken, then a beep says "your turn". Avoids talking over the translation.
    consecutive = bool(args.mic and player is not None) and (args.mode or "consecutive") == "consecutive"
    turn = {"waiting": False}

    async def finish_turn() -> None:
        try:
            await interp.flush()
            while player.is_speaking(tail_ms=300):
                await asyncio.sleep(0.05)
            await _play_cue(interp, args.output_device, None, beeps=1)
            print("  >>> your turn — speak the next sentence", file=sys.stderr)
        finally:
            turn["waiting"] = False

    async def consume() -> None:
        async for ev in interp.events():
            _print_event(ev, args.quiet)
            if (consecutive and ev.type == EventType.VAD_SPEECH_END and ev.data.get("reason") == "silence"
                    and not turn["waiting"] and interp.state == "running"):
                turn["waiting"] = True
                asyncio.create_task(finish_turn())
            if ev.type == EventType.TTS_AUDIO and ev.audio is not None:
                speech.extend([ev.audio, pause])
                if player is not None:
                    player.offer(ev)

    printer = asyncio.create_task(consume())
    capture_stats = None
    try:
        if args.mic:
            from ..capture import CONSENT_NOTICE, MicrophoneStream

            print(CONSENT_NOTICE, file=sys.stderr)
            cue = not args.no_cue
            if cue:   # played before capture opens, so the microphone does not hear it
                await _play_cue(interp, args.output_device, "Ready. Please speak Swahili now.")
            got, muted, muted_s = 0, False, 0.0
            mic = MicrophoneStream(args.device, sr, fallbacks=interp.fallbacks)
            async with mic:
                how = ("say one sentence, pause, listen to the English, speak again after the beep"
                       if consecutive else "one sentence at a time, pause ~1 s after each")
                print("\n" + "=" * 60 + f"\n  LISTENING for {args.seconds:.0f} s — speak Swahili now"
                      f"\n  ({how}; Ctrl+C to stop)\n" + "=" * 60, file=sys.stderr)
                async for chunk in mic:
                    # Echo guard: while English plays on speakers the mic would hear it and
                    # interpret it again. Push silence instead (keeps the timeline intact).
                    if turn["waiting"] or (player is not None and not args.no_echo_guard and player.is_speaking()):
                        if not muted:
                            print("  (mic closed — interpreting)", file=sys.stderr)
                        muted = True
                        muted_s += len(chunk) / sr
                        chunk = np.zeros_like(chunk)
                    else:
                        muted = False
                    await interp.push_audio(chunk, sr)
                    got += len(chunk)
                    if got >= args.seconds * sr:
                        break
            capture_stats = {**vars(mic.stats), "echo_guard_muted_s": round(muted_s, 2)}
            print("  (time is up — finishing what was heard)", file=sys.stderr)
            if cue:
                await _play_cue(interp, args.output_device, None, beeps=2)
        elif args.live_sim:
            from ..capture import stream_array

            async for chunk in stream_array(audio, sr, block_ms=100, realtime_factor=1.0):
                await interp.push_audio(chunk, sr)
        else:
            step = int(0.5 * sr)
            for i in range(0, len(audio), step):
                await interp.push_audio(audio[i:i + step], sr)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("stopped — finishing what was already heard", file=sys.stderr)

    summary = await interp.shutdown(drain=True)
    await printer
    if player is not None:
        await player.close()
        summary["playback"] = player.stats()
        units = {u["unit_id"]: u for u in interp.units()}
        lags = [t - units[k]["speech_end_t_ms"] for k, t in player.first_played_ms.items()
                if k in units and units[k].get("speech_end_t_ms") is not None]
        if lags:
            from ..utils.stats import summarize
            summary["latency_from_speech_end_ms"]["first_played"] = summarize(lags)
    summary["pacing"] = "realtime" if live else "batch (latencies include backlog; use --live-sim)"
    if capture_stats:
        summary["capture"] = capture_stats

    units = interp.units()
    english = " ".join(u["english"] for u in units if u["english"]).strip()
    print("\nEnglish:", english or "∅")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), file=sys.stderr)

    wav = np.concatenate(speech[:-1]) if speech else np.zeros(0, np.float32)
    if args.output and len(wav):
        print(f"wrote {save_wav(args.output, wav, out_sr)}", file=sys.stderr)
    if args.export:
        run_dir = new_run_dir(paths["output_dir"], "interpret")
        keep_text = cfg.logging.log_text
        if keep_text:
            (run_dir / "transcript_sw.txt").write_text(
                "\n".join(u["swahili"] for u in units) + "\n", encoding="utf-8")
            (run_dir / "translation_en.txt").write_text(
                "\n".join(u["english"] for u in units) + "\n", encoding="utf-8")
        rows = [u if keep_text else {**u, "swahili": None, "english": None} for u in units]
        (run_dir / "units.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
        if len(wav):
            save_wav(run_dir / "speech_en.wav", wav, out_sr)
        interp.recorder.export_jsonl(run_dir / "events.jsonl")
        interp.recorder.export_csv(run_dir / "events.csv")
        mode = "interpret-mic" if args.mic else "interpret-live-sim" if args.live_sim else "interpret-batch"
        write_manifest(run_dir, build_manifest(
            run_id=run_dir.name, mode=mode, config=cfg.model_dump(mode="json", by_alias=True),
            runtime=interp.runtime.to_dict(), fallbacks=interp.fallbacks,
            extra={"input": info.to_dict() if info else {"source": "microphone"}, "levels": levels,
                   "models": interp.model_status(), "summary": summary,
                   "text_logged": keep_text}))
        print(f"exported to {run_dir}", file=sys.stderr)
    return 3 if interp.state == "failed" else 0
