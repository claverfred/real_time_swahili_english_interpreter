"""Local speaker sink for ``tts_audio`` events, with playback metrics.

Chunks are played strictly in order on one worker thread. Measured per chunk:
* playback delay: ``tts_audio`` emitted -> chunk starts playing (waiting behind earlier speech);
* underrun: a later chunk of the same unit starts more than ``underrun_gap_ms`` after the
  previous one ended (synthesis did not keep up, so the listener hears a gap).
``cancel()`` drops queued chunks and stops the current one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..utils.stats import summarize
from .events import Event, EventRecorder


class AudioPlayer:
    def __init__(self, recorder: EventRecorder, *, device: int | str | None = None, pause_ms: int = 120,
                 underrun_gap_ms: float = 50.0, play_fn: Callable[[np.ndarray, int], None] | None = None,
                 stop_fn: Callable[[], None] | None = None) -> None:
        self.recorder, self.device = recorder, device
        self.pause_ms, self.underrun_gap_ms = pause_ms, underrun_gap_ms
        if play_fn is None:
            from ..audio.playback import play, stop

            def play_fn(audio: np.ndarray, sr: int) -> None:
                play(audio, sr, device=device, blocking=True)
            stop_fn = stop
        self._play, self._stop = play_fn, stop_fn
        self._q: asyncio.Queue[Event | None] = asyncio.Queue()
        self._ex = ThreadPoolExecutor(1, thread_name_prefix="swi-play")
        self._task: asyncio.Task[None] | None = None
        self.delays: list[float] = []
        self.underrun_gaps: list[float] = []
        self.first_played_ms: dict[Any, float] = {}   # unit_id -> recorder time playback started
        self.played_s = 0.0
        self._last: tuple[Any, float] | None = None   # (unit_id, end time)
        self._playing = False
        self._ended_ms = -1e18

    def is_speaking(self, tail_ms: float = 400.0) -> bool:
        """True while English audio is playing, queued, or ended less than ``tail_ms`` ago
        (room echo). Used to mute the microphone so the interpreter does not hear itself."""
        return self._playing or not self._q.empty() or self.recorder.now_ms() - self._ended_ms < tail_ms

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def offer(self, ev: Event) -> None:
        if ev.audio is not None:
            self._q.put_nowait(ev)

    def cancel(self) -> None:
        while not self._q.empty():
            self._q.get_nowait()
        if self._stop is not None:
            self._stop()

    async def close(self) -> None:
        self._q.put_nowait(None)
        if self._task is not None:
            await self._task
        self._ex.shutdown(wait=True)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            sr = int(ev.data.get("sample_rate", 22050))
            unit = ev.data.get("unit_id")
            audio = np.concatenate([ev.audio, np.zeros(int(sr * self.pause_ms / 1000), np.float32)])
            start = self.recorder.now_ms()
            self.delays.append(start - ev.t_ms)
            if self._last is not None and self._last[0] == unit and ev.data.get("chunk", 0) > 0:
                gap = start - self._last[1]
                if gap > self.underrun_gap_ms:
                    self.underrun_gaps.append(gap)
            self.first_played_ms.setdefault(unit, start)
            self._playing = True
            try:
                await loop.run_in_executor(self._ex, self._play, audio, sr)
            except Exception as exc:  # device unplugged etc.: report once per chunk, keep text flowing
                self.recorder.emit("error", stage="playback", error=f"{type(exc).__name__}: {exc}")
            finally:
                self._playing = False
                self._ended_ms = self.recorder.now_ms()
            self.played_s += len(ev.audio) / sr
            self._last = (unit, self._ended_ms)

    def stats(self) -> dict[str, Any]:
        return {"chunks": len(self.delays), "played_s": round(self.played_s, 3),
                "playback_delay_ms": summarize(self.delays) if self.delays else None,
                "underruns": len(self.underrun_gaps),
                "underrun_gap_ms": summarize(self.underrun_gaps) if self.underrun_gaps else None}
