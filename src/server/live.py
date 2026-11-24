"""One live interpretation session over a WebSocket (``/ws/live``).

Protocol. Text frames are JSON; binary frames are audio.

Client -> server
    {"type": "start", "sample_rate": 48000, "mode": "consecutive" | "simultaneous",
     "tts": true, "partials": false, "export": true}                       first message
    <binary>  little-endian float32 mono PCM at ``sample_rate`` (any block size)
    {"type": "listening"}   consecutive mode: the English has finished playing, mic is open again
    {"type": "flush" | "cancel" | "reset" | "stop" | "ping"}

Server -> client
    {"type": "loading"}                                      models are still loading
    {"type": "ready", "session_id", "mode", "models", ...}
    {"type": "event", "event": {...}}                        pipeline events (VAD, ASR, MT, latencies, errors)
    {"type": "tts_audio", "unit_id", "seq", "sample_rate", "samples"} then one binary frame of int16 PCM
    {"type": "turn", "state": "interpreting" | "spoken"}     consecutive mode turn-taking
    {"type": "status", "status": {...}}                      queues and counters, every 2 s
    {"type": "summary", "summary", "run_id", "files"}        after stop; then the socket closes
    {"type": "error", "code", "message"}

Turn-taking (consecutive mode) mirrors ``interpret --mic --play``: when the speaker pauses
(VAD end on silence) the server flushes the pipeline and ignores incoming audio (replaced by
silence, so the timeline stays intact) until the client reports ``listening``. The client
plays the English, beeps and reopens the mic. In simultaneous mode audio is always used; the
browser mutes its mic while English plays unless the user says they wear headphones.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from ..config.schema import InterpreterConfig
from ..pipeline import EventType, RealtimeSwahiliInterpreter
from .pool import ModelPool, PoolBusy

FORWARD = {EventType.VAD_SPEECH_START, EventType.VAD_SPEECH_END, EventType.ASR_PARTIAL, EventType.ASR_FINAL,
           EventType.MT_PARTIAL, EventType.MT_FINAL, EventType.PIPELINE_COMPLETE, EventType.FALLBACK,
           EventType.ERROR}

# WebSocket close codes (4000-4999 are application-defined)
CLOSE_BAD_REQUEST, CLOSE_UNAUTHORIZED, CLOSE_FORBIDDEN, CLOSE_TIMEOUT, CLOSE_BUSY, CLOSE_FAILED = (
    4400, 4401, 4403, 4408, 4409, 4500)


class ProtocolError(ValueError):
    pass


def parse_start(msg: dict[str, Any]) -> dict[str, Any]:
    if msg.get("type") != "start":
        raise ProtocolError("the first message must be {\"type\": \"start\", ...}")
    sr = msg.get("sample_rate", 16000)
    if not isinstance(sr, int) or not 8000 <= sr <= 192000:
        raise ProtocolError(f"sample_rate must be an integer 8000-192000, got {sr!r}")
    mode = msg.get("mode", "consecutive")
    if mode not in ("consecutive", "simultaneous"):
        raise ProtocolError(f"mode must be consecutive or simultaneous, got {mode!r}")
    tts = bool(msg.get("tts", True))
    return {"sample_rate": sr, "mode": mode if tts else "simultaneous", "tts": tts,
            "partials": bool(msg.get("partials", False)), "export": bool(msg.get("export", True))}


class LiveSession:
    def __init__(self, ws: WebSocket, pool: ModelPool, cfg: InterpreterConfig, *, output_dir: Path,
                 register_run: Callable[[Path], str], user_set_silence: bool = False) -> None:
        self.ws = ws
        self.pool = pool
        self.base_cfg = cfg
        self.scfg = cfg.server
        self.output_dir = output_dir
        self.register_run = register_run
        self.user_set_silence = user_set_silence
        self.interp: RealtimeSwahiliInterpreter | None = None
        self._send_lock = asyncio.Lock()
        self._waiting = False                  # consecutive mode: mic closed until the client says "listening"
        self._speech: list[np.ndarray] = []
        self.opts: dict[str, Any] = {}
        self.muted_s = 0.0

    # -- sending ---------------------------------------------------------------------------------
    async def send(self, obj: dict[str, Any]) -> None:
        if self.ws.client_state != WebSocketState.CONNECTED:
            return
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj, ensure_ascii=False, default=str))

    async def _send_audio(self, header: dict[str, Any], pcm: bytes) -> None:
        if self.ws.client_state != WebSocketState.CONNECTED:
            return
        async with self._send_lock:          # header and payload must stay adjacent
            await self.ws.send_text(json.dumps(header))
            await self.ws.send_bytes(pcm)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.ws.client_state == WebSocketState.CONNECTED:
            try:
                await self.ws.close(code, reason[:120])
            except RuntimeError:
                pass

    async def fail(self, code: int, err: str, message: str) -> None:
        try:
            await self.send({"type": "error", "code": err, "message": message})
        except Exception:
            pass
        await self.close(code, message)

    # -- session ---------------------------------------------------------------------------------
    async def run(self) -> None:
        try:
            raw = await asyncio.wait_for(self.ws.receive_text(), timeout=30)
            self.opts = parse_start(json.loads(raw))
        except (asyncio.TimeoutError, json.JSONDecodeError, ProtocolError, KeyError) as exc:
            await self.fail(CLOSE_BAD_REQUEST, "bad_request", str(exc) or "no start message")
            return
        except WebSocketDisconnect:
            return
        cfg = self.base_cfg.model_copy(deep=True)
        if self.opts["mode"] == "consecutive" and not self.user_set_silence:
            cfg.vad.min_silence_ms = self.scfg.consecutive_silence_ms
        if self.pool.state != "ready":
            await self.send({"type": "loading"})
        try:
            async with self.pool.lease("live session") as comps:
                await self._session(cfg, comps)
        except PoolBusy as exc:
            await self.fail(CLOSE_BUSY, "busy", str(exc))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            await self.fail(CLOSE_FAILED, "failed", f"{type(exc).__name__}: {exc}")

    async def _session(self, cfg: InterpreterConfig, comps: Any) -> None:
        o = self.opts
        interp = self.interp = RealtimeSwahiliInterpreter(cfg, components=comps, enable_tts=o["tts"],
                                                          partials=o["partials"], local_files_only=True)
        await interp.start()
        models = interp.model_status()
        await self.send({"type": "ready", "session_id": interp.recorder.session_id, "mode": o["mode"],
                         "tts": o["tts"], "pipeline_sample_rate": interp.sample_rate,
                         "tts_sample_rate": cfg.tts.sample_rate, "max_session_s": self.scfg.max_session_s,
                         "models": {k: models.get(k, {}).get("model") or models.get(k, {}).get("backend")
                                    for k in ("vad", "asr", "mt", "tts")}})
        pump = asyncio.create_task(self._pump_events())
        ticker = asyncio.create_task(self._status_ticker())
        stopped, reason = False, "client_stop"
        try:
            reason = await self._receive_loop()
            stopped = reason != "disconnect"
        finally:
            ticker.cancel()
            summary = await interp.shutdown(drain=stopped)
            await pump
        if not stopped:
            return
        summary["capture"] = {"source": "browser", "client_sample_rate": o["sample_rate"],
                              "turn_muted_s": round(self.muted_s, 2), "mode": o["mode"], "end": reason}
        out: dict[str, Any] = {"type": "summary", "summary": summary, "run_id": None, "files": []}
        if o["export"]:
            from ..pipeline.export import export_session

            sr = cfg.tts.sample_rate
            wav = np.concatenate(self._speech[:-1]) if self._speech else None
            run_dir = await asyncio.get_running_loop().run_in_executor(None, lambda: export_session(
                interp, self.output_dir, mode="interpret-browser", speech=wav, speech_sr=sr, summary=summary,
                extra={"input": {"source": "browser microphone", "sample_rate": o["sample_rate"]}}))
            out["run_id"] = self.register_run(run_dir)
            out["files"] = sorted(p.name for p in run_dir.iterdir())
        await self.send(out)
        await self.close(1000, reason)

    async def _receive_loop(self) -> str:
        """Returns why the session ended: client_stop, time_limit, idle_timeout or disconnect."""
        interp, o = self.interp, self.opts
        assert interp is not None
        deadline = time.monotonic() + self.scfg.max_session_s
        limit = self.scfg.max_ws_message_bytes
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                await self.send({"type": "error", "code": "time_limit",
                                 "message": f"session limit of {self.scfg.max_session_s:.0f} s reached"})
                return "time_limit"
            try:
                msg = await asyncio.wait_for(self.ws.receive(), timeout=min(self.scfg.idle_timeout_s, left))
            except asyncio.TimeoutError:
                if time.monotonic() >= deadline:
                    continue
                await self.send({"type": "error", "code": "idle_timeout",
                                 "message": f"no data for {self.scfg.idle_timeout_s:.0f} s"})
                return "idle_timeout"
            if msg["type"] == "websocket.disconnect":
                return "disconnect"
            data, text = msg.get("bytes"), msg.get("text")
            if data is not None:
                if len(data) > limit or len(data) % 4:
                    await self.send({"type": "error", "code": "bad_audio",
                                     "message": f"audio frames must be float32 and at most {limit} bytes"})
                    continue
                audio = np.nan_to_num(np.frombuffer(data, "<f4"), nan=0.0, posinf=0.0, neginf=0.0)
                audio = np.clip(audio, -1.0, 1.0)
                if self._waiting:
                    self.muted_s += len(audio) / o["sample_rate"]
                    audio = np.zeros_like(audio)
                await interp.push_audio(audio, o["sample_rate"])
                continue
            if text is None:
                continue
            if len(text) > limit:
                await self.send({"type": "error", "code": "bad_request", "message": "message too large"})
                continue
            try:
                kind = json.loads(text).get("type")
            except (json.JSONDecodeError, AttributeError):
                await self.send({"type": "error", "code": "bad_request", "message": "invalid JSON"})
                continue
            if kind == "stop":
                return "client_stop"
            if kind == "listening":
                self._waiting = False
            elif kind == "flush":
                asyncio.create_task(self._flush_and_report())
            elif kind == "cancel":
                await interp.cancel()
                self._waiting = False
                await self.send({"type": "turn", "state": "cancelled"})
            elif kind == "reset":
                await interp.reset()
                self._waiting = False
                await self.send({"type": "turn", "state": "cancelled"})
            elif kind == "ping":
                await self.send({"type": "pong"})
            else:
                await self.send({"type": "error", "code": "bad_request", "message": f"unknown message {kind!r}"})

    async def _flush_and_report(self) -> None:
        interp = self.interp
        try:
            if interp is not None and interp.state == "running":
                ok = await interp.flush()
                await self.send({"type": "turn", "state": "spoken" if ok else "cancelled"})
        except Exception:
            pass

    async def _end_turn(self) -> None:
        await self.send({"type": "turn", "state": "interpreting"})
        await self._flush_and_report()

    async def _pump_events(self) -> None:
        interp = self.interp
        assert interp is not None
        pause = np.zeros(int(interp.cfg.tts.sample_rate * interp.cfg.tts.sentence_pause_ms / 1000), np.float32)
        consecutive = self.opts["mode"] == "consecutive"
        async for ev in interp.events():
            try:
                if ev.type == EventType.TTS_AUDIO and ev.audio is not None:
                    audio = np.asarray(ev.audio, np.float32)
                    if self.opts["export"]:
                        self._speech.extend([audio, pause])
                    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    await self._send_audio({"type": "tts_audio", "unit_id": ev.data.get("unit_id"),
                                            "seq": ev.data.get("seq", ev.data.get("chunk")),
                                            "sample_rate": ev.data.get("sample_rate"),
                                            "samples": len(audio)}, pcm)
                elif ev.type in FORWARD:
                    await self.send({"type": "event", "event": ev.to_dict()})
                    if (consecutive and ev.type == EventType.VAD_SPEECH_END and ev.data.get("reason") == "silence"
                            and not self._waiting and interp.state == "running"):
                        self._waiting = True
                        asyncio.create_task(self._end_turn())
                elif ev.type == EventType.STATUS and ev.data.get("status") == "audio_overflow":
                    await self.send({"type": "event", "event": ev.to_dict()})
            except (WebSocketDisconnect, RuntimeError):
                pass                              # client gone; keep draining so shutdown completes

    async def _status_ticker(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            if self.interp is not None:
                try:
                    await self.send({"type": "status", "status": self.interp.status(),
                                     "health": self.interp.health()["status"], "waiting": self._waiting})
                except Exception:
                    return
