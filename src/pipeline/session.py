"""Synchronous wrapper around :class:`RealtimeSwahiliInterpreter` for notebooks and scripts.

Jupyter/Colab kernels already run an asyncio loop, so ``asyncio.run`` cannot be used in a cell.
The session runs the interpreter on its own event loop in a background thread and keeps the
models loaded between calls:

    with InterpreterSession("colab") as s:
        r = s.interpret("speech.wav")              # or a float32 array + sample_rate
        print(r["swahili"], r["english"]); Audio(r["audio"], rate=r["sample_rate"])

Each call goes through the same pipeline as the live interpreter (VAD, ASR, stabilization,
MT, TTS). ``realtime=True`` feeds the audio at speaking pace, so latencies are listener
latencies; otherwise it is fed as fast as possible (batch).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import numpy as np

from .events import EventType
from .interpreter import RealtimeSwahiliInterpreter


class InterpreterSession:
    def __init__(self, config: Any = None, *, overrides: list[str] | None = None, enable_tts: bool = True,
                 keep_context: bool = False, components: Any = None, start_timeout: float = 1800.0) -> None:
        self.keep_context = keep_context
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="interpreter-session", daemon=True)
        self._thread.start()
        self._audio: list[np.ndarray] = []
        self._events: list[Any] = []
        self.interp = RealtimeSwahiliInterpreter(config, overrides=list(overrides or []), enable_tts=enable_tts,
                                                 partials=False, components=components)
        t0 = time.perf_counter()
        self._call(self._start(), start_timeout)
        self.load_s = round(time.perf_counter() - t0, 1)

    # -- plumbing ----------------------------------------------------------------------------------
    def _call(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("session is closed")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _start(self) -> None:
        await self.interp.start()
        self._collector = asyncio.create_task(self._collect())

    async def _collect(self) -> None:
        async for ev in self.interp.events():
            if ev.type == EventType.TTS_AUDIO and ev.audio is not None:
                self._audio.append(np.asarray(ev.audio, dtype=np.float32).reshape(-1))
            elif ev.type in (EventType.ASR_FINAL, EventType.MT_FINAL, EventType.ERROR):
                self._events.append(ev)

    # -- API ---------------------------------------------------------------------------------------
    @property
    def cfg(self) -> Any:
        return self.interp.cfg

    def model_status(self) -> dict[str, Any]:
        return self.interp.model_status()

    def interpret(self, audio: str | Path | np.ndarray, sample_rate: int | None = None, *,
                  realtime: bool = False, block_s: float = 0.1, timeout: float | None = 600.0) -> dict[str, Any]:
        """Interpret one recording. Returns Swahili, English, per-unit latencies, English audio."""
        if isinstance(audio, (str, Path)):
            from ..audio import load_audio

            audio, _ = load_audio(str(audio), self.interp.sample_rate, self.cfg.audio.max_input_seconds,
                                  self.cfg.audio.max_input_mb, self.interp.fallbacks)
            sample_rate = self.interp.sample_rate
        if sample_rate is None:
            raise ValueError("sample_rate is required for array input")
        return self._call(self._interpret(np.asarray(audio), int(sample_rate), realtime, block_s), timeout)

    async def _interpret(self, audio: np.ndarray, sr: int, realtime: bool, block_s: float) -> dict[str, Any]:
        n0 = len(self.interp.units())
        self._audio.clear()
        self._events.clear()
        step = max(1, int(sr * block_s)) if realtime else sr
        t0 = time.perf_counter()
        for k, i in enumerate(range(0, len(audio), step)):
            await self.interp.push_audio(audio[i:i + step], sr)
            if realtime:
                lag = t0 + (k + 1) * block_s - time.perf_counter()
                if lag > 0:
                    await asyncio.sleep(lag)
        await self.interp.flush()
        await asyncio.sleep(0.05)                       # let the collector take the last events
        wall = time.perf_counter() - t0
        units = self.interp.units()[n0:]
        out_sr = self.cfg.tts.sample_rate
        pause = np.zeros(int(out_sr * self.cfg.tts.sentence_pause_ms / 1000), np.float32)
        parts: list[np.ndarray] = []
        for a in self._audio:
            parts += [a, pause]
        errors = [str(e.data.get("error") or e.data)[:300] for e in self._events if e.type == EventType.ERROR]
        if not self.keep_context:
            await self.interp.reset()
        return {
            "swahili": " ".join(u["swahili"] for u in units if u.get("swahili")).strip(),
            "english": " ".join(u["english"] for u in units if u.get("english")).strip(),
            "units": units, "audio": np.concatenate(parts) if parts else np.zeros(0, np.float32),
            "sample_rate": out_sr, "input_s": round(len(audio) / sr, 2), "wall_s": round(wall, 2),
            "realtime": realtime, "errors": errors,
        }

    async def _shutdown(self) -> dict[str, Any]:
        summary = await self.interp.shutdown(drain=False)
        await self._collector
        return summary

    def reset(self) -> None:
        self._call(self.interp.reset())

    def close(self) -> dict[str, Any] | None:
        if not self._thread.is_alive():
            return None
        try:
            summary = self._call(self._shutdown(), 120)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
        return summary

    def __enter__(self) -> InterpreterSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
