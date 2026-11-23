"""Async real-time pipeline and the public streaming API.

    interpreter = RealtimeSwahiliInterpreter("cpu")
    await interpreter.start()
    await interpreter.push_audio(chunk, sample_rate=16000)
    async for event in interpreter.events():
        if event.type == "mt_final":
            print("English:", event.text)

Architecture (producer-consumer)::

    push_audio -> [audio_q] -> VAD+ASR worker -> [mt_q] -> stabilize+MT worker -> [tts_q] -> TTS worker
                                                                                      |
                                       every stage emits timestamped events -> events()

* Each stage runs blocking model code in its **own single-thread executor**: stages overlap
  (ASR decodes the next utterance while the previous one is translated and spoken), and each
  stage's state is only ever touched by one thread.
* Queues are bounded (``streaming.queue_max``): a slow stage applies backpressure upstream
  instead of growing memory. ``push_audio(block=False)`` drops and counts instead of waiting.
* The ASR worker drains every chunk that queued up while it was busy and processes them in one
  pass, so a backlog costs one VAD pass, not one ASR call per chunk. The MT worker skips ASR
  partials that a newer update of the same segment already superseded.
* ``flush()`` sends a marker through all stages and returns when everything before it is
  spoken. ``cancel()`` bumps a generation counter: queued and in-flight work from before is
  dropped, and speech stops at the next chunk boundary.
* A failure on one item (e.g. TTS after all fallbacks) emits an ``error`` event and the
  pipeline continues; the text results are still delivered.

Latency is measured from the moment the audio containing the end of the utterance was
pushed (``speech end``) to ASR final, MT final, first English audio and full English audio.
With real-time pacing (microphone, ``--live-sim``) these are what a listener experiences.
"""

from __future__ import annotations

import asyncio
import bisect
import time
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..config import load_config
from ..config.schema import InterpreterConfig
from ..platform.resolve import resolve_runtime
from ..utils.stats import summarize
from .events import Event, EventRecorder, EventType

State = Literal["idle", "loading", "running", "stopping", "stopped", "failed"]
STAGES = ("asr", "mt", "tts")


@dataclass
class Components:
    """Pre-built stage components (tests, or sharing loaded models between sessions)."""
    vad: Any = None
    asr: Any = None
    translator: Any = None
    synthesizer: Any = None


@dataclass
class _Marker:
    kind: Literal["flush", "stop"]
    future: asyncio.Future[bool] | None = None


@dataclass
class _Item:
    payload: Any
    t_ms: float          # enqueue time (recorder clock)
    gen: int             # cancel generation
    extra: dict[str, Any] = field(default_factory=dict)


def _to_mono_float(chunk: Any) -> np.ndarray:
    a = np.asarray(chunk)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    elif a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = a.mean(axis=1 if a.shape[1] <= 2 else 0)
    return np.ascontiguousarray(a.reshape(-1))


class RealtimeSwahiliInterpreter:
    def __init__(self, config: InterpreterConfig | str | None = None, *, overrides: list[str] | None = None,
                 components: Components | None = None, recorder: EventRecorder | None = None,
                 enable_tts: bool = True, partials: bool | None = None, local_files_only: bool = False,
                 event_buffer: int = 8192) -> None:
        if config is None or isinstance(config, str):
            config = load_config(config, overrides)
        self.cfg, self.runtime, self.fallbacks = resolve_runtime(config)
        self.sample_rate = self.cfg.audio.sample_rate
        self.recorder = recorder or EventRecorder(log_text=self.cfg.logging.log_text)
        self.components = components or Components()
        self.enable_tts = enable_tts
        self.partials = self.cfg.asr.partials if partials is None else partials
        self.local_files_only = local_files_only
        self._event_buffer = event_buffer
        self.state: State = "idle"
        self._events_q: asyncio.Queue[Event | None] | None = None
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._ex: dict[str, ThreadPoolExecutor] = {}
        self._gen = 0
        self._models: dict[str, dict[str, Any]] = {}
        self._errors: list[dict[str, Any]] = []
        self._units: list[dict[str, Any]] = []
        self._asr_final_t: dict[int, float] = {}
        self._last_mt_partial: dict[int, str] = {}
        self._arrivals: list[tuple[int, float]] = []   # (VAD samples processed so far, push time ms)
        self._vad_samples = 0
        self._qwait: dict[str, deque[float]] = {s: deque(maxlen=10000) for s in ("push", *STAGES)}
        self._stage_ms: dict[str, deque[float]] = {k: deque(maxlen=10000) for k in ("asr_final", "asr_partial", "mt", "tts_chunk")}
        self.counters: dict[str, int] = dict.fromkeys(
            ["chunks_in", "samples_in", "overflows", "asr_partials", "asr_finals", "asr_revisions",
             "mt_partials", "mt_finals", "mt_skipped", "mt_revisions", "stale_partials_skipped",
             "tts_chunks", "tts_audio_samples", "units_complete", "cancels", "errors"], 0)
        self._t_start: float | None = None
        self._pending_flush: set[asyncio.Future[bool]] = set()

    # ==== lifecycle =============================================================================
    async def start(self) -> None:
        if self.state not in ("idle", "stopped"):
            raise RuntimeError(f"cannot start from state {self.state!r}")
        loop = asyncio.get_running_loop()
        self._events_q = self.recorder.subscribe(self._event_buffer)  # before anything is emitted
        self.state = "loading"
        self._ex = {s: ThreadPoolExecutor(1, thread_name_prefix=f"swi-{s}") for s in STAGES}
        self.recorder.emit(EventType.STATUS, status="loading", profile=self.cfg.profile,
                           device=self.runtime.device)
        try:
            # Each stage loads in the thread that will run it (thread-affine backends such as SAPI).
            loads = [loop.run_in_executor(self._ex["asr"], self._load_asr),
                     loop.run_in_executor(self._ex["mt"], self._load_mt)]
            if self.enable_tts:
                loads.append(loop.run_in_executor(self._ex["tts"], self._load_tts))
            await asyncio.gather(*loads)
        except BaseException as exc:
            self.state = "failed"
            self._record_error("load", exc)
            self._shutdown_executors()
            raise
        for rec in self.fallbacks.records:
            self.recorder.emit(EventType.FALLBACK, stage=rec.stage, requested=rec.requested,
                               selected=rec.selected, reason=rec.reason)
        maxq = self.cfg.streaming.queue_max
        self._audio_q: asyncio.Queue[_Item] = asyncio.Queue(maxq)
        self._mt_q: asyncio.Queue[_Item] = asyncio.Queue(maxq)
        self._tts_q: asyncio.Queue[_Item] = asyncio.Queue(maxq)
        self._tasks = {"asr": asyncio.create_task(self._guard("asr", self._asr_worker())),
                       "mt": asyncio.create_task(self._guard("mt", self._mt_worker())),
                       "tts": asyncio.create_task(self._guard("tts", self._tts_worker()))}
        self._t_start = time.perf_counter()
        self.state = "running"
        self.recorder.emit(EventType.STATUS, status="running", models=self.model_status())

    async def push_audio(self, chunk: Any, sample_rate: int | None = None, *, block: bool = True) -> bool:
        """Queue audio (float32 [-1, 1] or int16, mono or stereo, any rate). Returns False if it
        was dropped because the pipeline is full and ``block=False``."""
        self._require_running()
        audio = _to_mono_float(chunk)
        sr = sample_rate or self.sample_rate
        if sr != self.sample_rate:
            from ..audio import resample
            audio = resample(audio, sr, self.sample_rate)
        if not len(audio):
            return True
        item = _Item(audio, self.recorder.now_ms(), self._gen)
        if block:
            await self._audio_q.put(item)
            self._qwait["push"].append(self.recorder.now_ms() - item.t_ms)   # backpressure felt by the source
        else:
            try:
                self._audio_q.put_nowait(item)
            except asyncio.QueueFull:
                self.counters["overflows"] += 1
                self.recorder.emit(EventType.STATUS, status="audio_overflow", samples=len(audio),
                                   overflows=self.counters["overflows"])
                return False
        self.counters["chunks_in"] += 1
        self.counters["samples_in"] += len(audio)
        return True

    async def flush(self, timeout: float | None = None) -> bool:
        """End the current utterance and wait until everything pushed so far is translated and
        spoken. Returns False if a cancel discarded it."""
        self._require_running()
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending_flush.add(fut)
        fut.add_done_callback(self._pending_flush.discard)
        await self._audio_q.put(_Item(_Marker("flush", fut), self.recorder.now_ms(), self._gen))
        return await asyncio.wait_for(asyncio.shield(fut), timeout)

    async def cancel(self) -> None:
        """Drop queued and in-flight work and stop speaking. Translation context is kept."""
        self._require_running()
        await self._cancel(full_reset=False)

    async def reset(self) -> None:
        """Cancel and forget conversation context (prompt and translation history)."""
        self._require_running()
        await self._cancel(full_reset=True)

    async def shutdown(self, drain: bool = True, timeout: float | None = None) -> dict[str, Any]:
        """Stop the session. With ``drain`` everything already pushed is finished first.
        Returns the session summary (also emitted as a ``status`` event)."""
        if self.state == "failed":       # release threads and end events(); keep the failed state
            for t in self._tasks.values():
                t.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            await asyncio.get_running_loop().run_in_executor(None, self._shutdown_executors)
            self.recorder.close()
            return self.summary()
        if self.state != "running":
            if self.state in ("stopped", "idle"):
                return self.summary()
            raise RuntimeError(f"cannot shut down from state {self.state!r}")
        if drain:
            await self.flush(timeout)
        self.state = "stopping"
        await self._audio_q.put(_Item(_Marker("stop"), self.recorder.now_ms(), self._gen))
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        await asyncio.get_running_loop().run_in_executor(None, self._shutdown_executors)
        self.state = "stopped"
        summary = self.summary()
        self.recorder.emit(EventType.STATUS, status="stopped", summary=summary)
        self.recorder.close()
        return summary

    async def __aenter__(self) -> RealtimeSwahiliInterpreter:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.state == "running":
            await self.shutdown(drain=exc_type is None)

    async def say(self, text: str) -> tuple[np.ndarray, int] | None:
        """Synthesize a short English prompt (e.g. a UI cue) outside the pipeline queues.
        Returns (audio, sample_rate), or None when TTS is disabled or fails."""
        if getattr(self, "_synth", None) is None:
            return None
        try:
            res = await self._run("tts", self._synth.synthesize, text)
        except Exception as exc:
            self._record_error("tts", exc, purpose="cue")
            return None
        return res.audio, res.sample_rate

    # ==== consumption ===========================================================================
    async def events(self) -> AsyncIterator[Event]:
        """All events from :meth:`start` on, in order, until shutdown. Single consumer; use
        ``recorder.stream()`` for additional live subscribers."""
        if self._events_q is None:
            self._events_q = self.recorder.subscribe(self._event_buffer)
        q = self._events_q
        while True:
            ev = await q.get()
            if ev is None:
                return
            yield ev

    # ==== introspection =========================================================================
    def status(self) -> dict[str, Any]:
        q = getattr(self, "_audio_q", None)
        return {
            "state": self.state,
            "session_id": self.recorder.session_id,
            "uptime_s": round(time.perf_counter() - self._t_start, 3) if self._t_start else 0.0,
            "audio_in_s": round(self.counters["samples_in"] / self.sample_rate, 3),
            "queues": {"audio": q.qsize() if q else 0,
                       "mt": self._mt_q.qsize() if q else 0,
                       "tts": self._tts_q.qsize() if q else 0,
                       "max": self.cfg.streaming.queue_max},
            "counters": dict(self.counters),
            "events_dropped": self.recorder.dropped,
            "tts_backend": self._synth.backend.name if getattr(self, "_synth", None) else None,
            "fallbacks": len(self.fallbacks.records),
            "last_error": self._errors[-1] if self._errors else None,
        }

    def health(self) -> dict[str, Any]:
        workers = {s: (t is not None and not t.done()) for s, t in
                   ((s, self._tasks.get(s)) for s in STAGES)}
        if self.state == "failed" or (self.state == "running" and not all(workers.values())):
            status = "failed"
        elif self.state == "running" and (self._errors or self.fallbacks.records):
            status = "degraded"
        elif self.state == "running":
            status = "ok"
        else:
            status = self.state
        return {"status": status, "state": self.state, "workers": workers,
                "errors": len(self._errors), "fallbacks": len(self.fallbacks.records),
                "last_error": self._errors[-1] if self._errors else None}

    def model_status(self) -> dict[str, Any]:
        out = {k: dict(v) for k, v in self._models.items()}
        if getattr(self, "_synth", None) is not None and "tts" in out:
            out["tts"]["active_backend"] = self._synth.backend.name
            out["tts"]["switches"] = list(self._synth.switches)
        out["runtime"] = self.runtime.to_dict()
        out["fallbacks"] = self.fallbacks.to_list()
        return out

    def units(self) -> list[dict[str, Any]]:
        """Per translation unit: Swahili, English, timing and latencies."""
        return [dict(u) for u in self._units]

    def summary(self) -> dict[str, Any]:
        def lat(key: str) -> dict[str, float] | None:
            vals = [u[key] for u in self._units if u.get(key) is not None]
            return summarize(vals) if vals else None

        try:
            import psutil
            rss_mb = round(psutil.Process().memory_info().rss / 2**20, 1)
        except Exception:  # pragma: no cover - psutil is a core dependency
            rss_mb = None
        audio_in = self.counters["samples_in"] / self.sample_rate
        return {
            "profile": self.cfg.profile, "device": self.runtime.device,
            "audio_in_s": round(audio_in, 3),
            "wall_s": round(time.perf_counter() - self._t_start, 3) if self._t_start else 0.0,
            "units": len(self._units), "counters": dict(self.counters),
            "latency_from_speech_end_ms": {"asr_final": lat("lat_asr_final_ms"), "mt_final": lat("lat_mt_final_ms"),
                                           "first_audio": lat("lat_first_audio_ms"),
                                           "full_audio": lat("lat_full_audio_ms")},
            "stage_ms": {k: summarize(list(v)) if v else None for k, v in self._stage_ms.items()},
            "queue_wait_ms": {k: summarize(list(v)) if v else None for k, v in self._qwait.items()},
            "events_dropped": self.recorder.dropped, "errors": list(self._errors[-20:]),
            "fallbacks": self.fallbacks.to_list(), "rss_mb": rss_mb,
        }

    # ==== loading (runs in the stage threads) ===================================================
    def _load_asr(self) -> None:
        from ..asr import create_asr
        from ..platform.storage import prepare_storage
        from ..vad import create_vad

        c, cfg = self.components, self.cfg
        t0 = time.perf_counter()
        vad = c.vad
        if vad is None:
            paths = prepare_storage(cfg)
            vad = create_vad(cfg.vad, self.sample_rate, cfg.audio.frame_ms, str(paths["cache_dir"]), self.fallbacks,
                             allow_fallback=cfg.fallback.policy != "fail")
        self._models["vad"] = {"backend": vad.name, "load_ms": round((time.perf_counter() - t0) * 1000, 1)}
        asr = c.asr or create_asr(cfg, self.fallbacks, local_files_only=self.local_files_only)
        info: dict[str, Any] = {"model": asr.model_name, "device": self.runtime.device,
                                "compute_type": cfg.asr.compute_type, **dict(getattr(asr, "load_info", {}) or {})}
        if hasattr(asr, "warmup"):
            info["first_inference_ms"] = round(asr.warmup(), 1)
            info["second_inference_ms"] = round(asr.warmup(), 1)
        self._models["asr"] = info
        from ..asr import StreamingTranscriber
        from ..vad import VadSegmenter

        self._segmenter = VadSegmenter(vad, cfg.vad, self.recorder)
        self._streamer = StreamingTranscriber(asr, cfg.asr, self.sample_rate, self.recorder, partials=self.partials)

    def _load_mt(self) -> None:
        from ..mt import MtStage, create_translator

        tr = self.components.translator or create_translator(self.cfg, self.fallbacks,
                                                             local_files_only=self.local_files_only)
        info: dict[str, Any] = {"model": tr.model_name, "license": self.cfg.mt.license,
                                "compute_type": self.cfg.mt.compute_type, **dict(tr.load_info or {})}
        info["first_inference_ms"] = round(tr.warmup(), 1)
        info["second_inference_ms"] = round(tr.warmup(), 1)
        self._models["mt"] = info
        self._mt = MtStage(tr, self.cfg.mt, self.cfg.text, self.recorder)

    def _load_tts(self) -> None:
        from ..tts import TtsStage, create_synthesizer

        syn = self.components.synthesizer or create_synthesizer(self.cfg, self.fallbacks)
        info: dict[str, Any] = {"backend": syn.backend.name, "voice": self.cfg.tts.voice,
                                "license": getattr(syn.backend, "license", None), "cloud": syn.backend.is_cloud,
                                "sample_rate": syn.sample_rate, **dict(syn.load_info or {})}
        try:  # non-fatal: a cloud voice may be briefly unreachable; runtime fallback handles it
            info["first_inference_ms"] = round(syn.warmup(), 1)
            info["second_inference_ms"] = round(syn.warmup(), 1)
        except Exception as exc:
            info["warmup_error"] = f"{type(exc).__name__}: {exc}"
        self._models["tts"] = info
        self._synth = syn
        self._tts = TtsStage(syn, self.recorder)

    # ==== workers ===============================================================================
    async def _guard(self, stage: str, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # a worker died: the session cannot continue
            self._record_error(stage, exc, fatal=True)
            self.state = "failed"
            for fut in self._pending_flush:
                if not fut.done():
                    fut.set_exception(RuntimeError(f"{stage} worker failed: {exc}"))
            for name, task in self._tasks.items():
                if name != stage:
                    task.cancel()

    async def _run(self, stage: str, fn: Any, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._ex[stage], fn, *args)

    def _wait(self, stage: str, item: _Item) -> None:
        self._qwait[stage].append(self.recorder.now_ms() - item.t_ms)

    async def _asr_worker(self) -> None:
        while True:
            items = [await self._audio_q.get()]
            while not isinstance(items[-1].payload, _Marker):
                try:
                    items.append(self._audio_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            marker = items[-1].payload if isinstance(items[-1].payload, _Marker) else None
            live = [i for i in items if not isinstance(i.payload, _Marker) and i.gen == self._gen]
            if live:
                gen = live[0].gen
                for it in live:
                    self._wait("asr", it)
                    self._vad_samples += len(it.payload)
                    self._arrivals.append((self._vad_samples, it.t_ms))
                audio = live[0].payload if len(live) == 1 else np.concatenate([i.payload for i in live])
                await self._forward_asr(await self._safe("asr", self._asr_process, audio), gen)
            if marker is None:
                continue
            if marker.kind == "flush":
                await self._forward_asr(await self._safe("asr", self._asr_flush), self._gen)
            await self._mt_q.put(_Item(marker, self.recorder.now_ms(), self._gen))
            if marker.kind == "stop":
                return

    def _asr_events(self, vad_events: list[Any]) -> list[Any]:
        """Decode per VAD event; a failing decode loses only its own segment, not the batch."""
        out: list[Any] = []
        for ev in vad_events:
            try:
                out += self._streamer.on_vad_event(ev)
            except Exception as exc:
                out.append(("error", exc, ev.segment_id))
        return out

    def _asr_process(self, audio: np.ndarray) -> list[Any]:
        return self._asr_events(self._segmenter.process(audio))

    def _asr_flush(self) -> list[Any]:
        return self._asr_events(self._segmenter.flush())

    async def _forward_asr(self, updates: list[Any] | None, gen: int) -> None:
        for u in updates or []:
            if isinstance(u, tuple):            # ("error", exc, segment_id) from _asr_events
                self._record_error("asr", u[1], segment_id=u[2])
                continue
            if u.kind == "partial":
                self.counters["asr_partials"] += 1
                self._stage_ms["asr_partial"].append(u.latency_ms)
            else:
                self.counters["asr_finals"] += 1
                self.counters["asr_revisions"] += u.revisions
                self._stage_ms["asr_final"].append(u.latency_ms)
                self._asr_final_t[u.segment_id] = self.recorder.now_ms()
            await self._mt_q.put(_Item(u, self.recorder.now_ms(), gen))

    async def _mt_worker(self) -> None:
        while True:
            items = [await self._mt_q.get()]
            while not isinstance(items[-1].payload, _Marker):
                try:
                    items.append(self._mt_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            for idx, it in enumerate(items):
                if isinstance(it.payload, _Marker):
                    m = it.payload
                    if m.kind == "flush":
                        end_s = self._vad_samples / self.sample_rate
                        await self._forward_mt(await self._safe("mt", self._mt.flush, end_s), self._gen)
                    await self._tts_q.put(_Item(m, self.recorder.now_ms(), self._gen))
                    if m.kind == "stop":
                        return
                    continue
                if it.gen != self._gen:
                    continue
                self._wait("mt", it)
                up = it.payload
                if up.kind == "partial" and any(not isinstance(n.payload, _Marker) and n.payload.segment_id == up.segment_id
                                                for n in items[idx + 1:]):
                    self.counters["stale_partials_skipped"] += 1   # a newer update already queued
                    continue
                await self._forward_mt(await self._safe("mt", self._mt.on_asr_update, up), it.gen)

    async def _forward_mt(self, updates: list[Any] | None, gen: int) -> None:
        for u in updates or []:
            if u.kind == "partial":
                self.counters["mt_partials"] += 1
                if u.segment_id is not None:
                    self._last_mt_partial[u.segment_id] = u.text
                continue
            if u.kind == "skipped":
                self.counters["mt_skipped"] += 1
                continue
            self.counters["mt_finals"] += 1
            self._stage_ms["mt"].append(u.result.latency_ms)
            segs = u.unit.segment_ids if u.unit else []
            shown = [self._last_mt_partial.pop(s) for s in segs if s in self._last_mt_partial]
            if shown and shown[-1] != u.text:
                self.counters["mt_revisions"] += 1         # a provisional translation was replaced
            rec = self._new_unit(u)
            if self.enable_tts and u.text:
                await self._tts_q.put(_Item(u, self.recorder.now_ms(), gen, {"unit": rec}))
            else:
                self._complete_unit(rec)

    async def _tts_worker(self) -> None:
        while True:
            it = await self._tts_q.get()
            if isinstance(it.payload, _Marker):
                if it.payload.future is not None and not it.payload.future.done():
                    it.payload.future.set_result(True)
                if it.payload.kind == "stop":
                    return
                continue
            if it.gen != self._gen:
                continue
            self._wait("tts", it)
            up, rec = it.payload, it.extra["unit"]
            speech = self._tts.speak(up.text, unit_id=rec["unit_id"], segment_id=up.segment_id)
            while True:
                if it.gen != self._gen:            # cancelled mid-unit: stop at a chunk boundary
                    speech.close()
                    rec["cancelled"] = True
                    break
                try:
                    tu = await self._run("tts", next, speech, None)
                except Exception as exc:
                    self._record_error("tts", exc, unit_id=rec["unit_id"], segment_id=up.segment_id)
                    rec["tts_error"] = str(exc)
                    break
                if tu is None:
                    break
                now = self.recorder.now_ms()
                self.counters["tts_chunks"] += 1
                self.counters["tts_audio_samples"] += len(tu.chunk.audio)
                self._stage_ms["tts_chunk"].append(tu.chunk.synth_ms)
                rec["audio_s"] = round(rec.get("audio_s", 0.0) + tu.chunk.duration_s, 3)
                rec["tts_backend"] = tu.chunk.backend
                if tu.is_first:
                    rec["lat_first_audio_ms"] = self._since_speech_end(rec, now)
                if tu.is_last:
                    rec["lat_full_audio_ms"] = self._since_speech_end(rec, now)
            if not rec.get("cancelled"):
                self._complete_unit(rec)

    # ==== unit bookkeeping ======================================================================
    def _speech_end_ms(self, end_s: float) -> float | None:
        """Push time of the audio that contained ``end_s`` (stream seconds)."""
        if not self._arrivals:
            return None
        target = int(end_s * self.sample_rate)
        i = bisect.bisect_left(self._arrivals, (target, -1.0))
        return self._arrivals[min(i, len(self._arrivals) - 1)][1]

    @staticmethod
    def _since_speech_end(rec: dict[str, Any], now: float) -> float | None:
        t = rec.get("speech_end_t_ms")
        return round(now - t, 1) if t is not None else None

    def _new_unit(self, u: Any) -> dict[str, Any]:
        unit, now = u.unit, self.recorder.now_ms()
        # VAD segment ends include hangover_ms after the last speech frame; measure from the last
        # speech frame instead (conservative by up to hangover_ms for force-split segments).
        speech_end_s = max(unit.start_s, unit.end_s - self.cfg.vad.hangover_ms / 1000) if unit else 0.0
        end_t = self._speech_end_ms(speech_end_s) if unit else None
        segs = list(unit.segment_ids) if unit else []
        asr_t = self._asr_final_t.get(segs[-1]) if segs else None
        rec: dict[str, Any] = {
            "unit_id": unit.unit_id if unit else None, "segment_ids": segs,
            "start_s": round(unit.start_s, 3) if unit else None, "end_s": round(unit.end_s, 3) if unit else None,
            "swahili": u.source, "english": u.text, "uncertain": bool(unit.uncertain) if unit else False,
            "flags": list(u.result.flags), "mt_ms": u.result.latency_ms, "speech_end_t_ms": end_t,
            "lat_asr_final_ms": round(asr_t - end_t, 1) if asr_t is not None and end_t is not None else None,
            "lat_mt_final_ms": round(now - end_t, 1) if end_t is not None else None,
        }
        self._units.append(rec)
        return rec

    def _complete_unit(self, rec: dict[str, Any]) -> None:
        self.counters["units_complete"] += 1
        self.recorder.emit(EventType.PIPELINE_COMPLETE, segment_id=rec["segment_ids"][-1] if rec["segment_ids"] else None,
                           text=rec["english"], source=rec["swahili"], unit_id=rec["unit_id"],
                           **{k: rec.get(k) for k in ("lat_asr_final_ms", "lat_mt_final_ms", "lat_first_audio_ms",
                                                      "lat_full_audio_ms", "audio_s", "tts_backend", "tts_error")})

    # ==== cancel / errors =======================================================================
    async def _cancel(self, full_reset: bool) -> None:
        self._gen += 1
        self.counters["cancels"] += 1
        for q in (self._audio_q, self._mt_q, self._tts_q):
            keep: list[_Item] = []
            while True:
                try:
                    it = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(it.payload, _Marker):
                    if it.payload.kind == "stop":
                        keep.append(it)
                    elif it.payload.future is not None and not it.payload.future.done():
                        it.payload.future.set_result(False)
            for it in keep:
                q.put_nowait(it)

        def asr_reset() -> None:
            self._segmenter.discard()
            if full_reset:
                self._streamer.reset()
            else:
                self._streamer.discard()

        # Serialized behind any in-flight call on each stage thread.
        await self._run("asr", asr_reset)
        await self._run("mt", self._mt.reset if full_reset else self._mt.cancel)
        self._last_mt_partial.clear()
        self.recorder.emit(EventType.STATUS, status="reset" if full_reset else "cancelled", generation=self._gen)

    async def _safe(self, stage: str, fn: Any, *args: Any) -> Any:
        try:
            return await self._run(stage, fn, *args)
        except Exception as exc:  # one bad item must not stop the session
            self._record_error(stage, exc)
            return None

    def _record_error(self, stage: str, exc: BaseException, fatal: bool = False, **kw: Any) -> None:
        err = {"stage": stage, "error": f"{type(exc).__name__}: {exc}", "fatal": fatal, **kw}
        self._errors.append(err)
        self.counters["errors"] += 1
        self.recorder.emit(EventType.ERROR, **err)

    def _require_running(self) -> None:
        if self.state != "running":
            raise RuntimeError(f"interpreter is {self.state}; call start() first")

    def _shutdown_executors(self) -> None:
        for ex in self._ex.values():
            ex.shutdown(wait=True)
