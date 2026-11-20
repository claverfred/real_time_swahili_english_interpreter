"""Timestamped pipeline events and the recorder that traces them.

Every stage emits an :class:`Event` with a monotonically increasing ``seq`` and a session-relative
monotonic timestamp, so stage latencies can be computed exactly and exported as JSONL/CSV.
"""

from __future__ import annotations

import asyncio
import csv
import itertools
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    AUDIO_FRAME_RECEIVED = "audio_frame_received"
    VAD_SPEECH_START = "vad_speech_start"
    VAD_SPEECH_END = "vad_speech_end"
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    TRANSCRIPT_STABILIZED = "transcript_stabilized"
    MT_REQUESTED = "mt_requested"
    MT_PARTIAL = "mt_partial"
    MT_FINAL = "mt_final"
    TTS_CHUNK_REQUESTED = "tts_chunk_requested"
    TTS_FIRST_AUDIO = "tts_first_audio"
    TTS_AUDIO = "tts_audio"
    TTS_CHUNK_COMPLETE = "tts_chunk_complete"
    TTS_COMPLETE = "tts_complete"
    PIPELINE_COMPLETE = "pipeline_complete"
    FALLBACK = "fallback"
    ERROR = "error"
    STATUS = "status"


# Event data keys that carry speech content; dropped from persisted logs when log_text=False.
TEXT_DATA_KEYS = frozenset({"source", "original", "stable", "unstable", "reference", "hypothesis"})


@dataclass
class Event:
    type: EventType
    session_id: str
    seq: int
    t_ms: float                      # monotonic ms since session start
    wall_time: float                 # unix epoch seconds
    segment_id: int | None = None
    text: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    audio: Any = None                # raw audio payload; never serialized

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("audio", None)
        d["type"] = self.type.value
        return d


class EventRecorder:
    """Thread-safe event log with async fan-out to subscribers.

    ``log_text=False`` drops transcript/translation text from the persisted log (privacy),
    while still delivering it to live subscribers.
    """

    def __init__(
        self,
        session_id: str | None = None,
        jsonl_path: Path | None = None,
        log_text: bool = True,
        keep_in_memory: bool = True,
        subscriber_queue_max: int = 1024,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._t0 = time.perf_counter()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._events: list[Event] = []
        self._keep = keep_in_memory
        self._log_text = log_text
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[Event | None]]] = []
        self._sub_max = subscriber_queue_max
        self._jsonl = None
        if jsonl_path is not None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = jsonl_path.open("a", encoding="utf-8")

    # -- emission -----------------------------------------------------------------------------
    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def emit(
        self,
        type: EventType | str,
        *,
        segment_id: int | None = None,
        text: str | None = None,
        audio: Any = None,
        **data: Any,
    ) -> Event:
        etype = EventType(type)
        with self._lock:
            ev = Event(
                type=etype,
                session_id=self.session_id,
                seq=next(self._seq),
                t_ms=round(self.now_ms(), 3),
                wall_time=time.time(),
                segment_id=segment_id,
                text=text,
                data=data,
                audio=audio,
            )
            if self._keep:
                self._events.append(ev)
            if self._jsonl is not None:
                rec = self._persisted(ev)
                self._jsonl.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                self._jsonl.flush()
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(self._offer, queue, ev)
        return ev

    @staticmethod
    def _offer(queue: asyncio.Queue[Event | None], ev: Event) -> None:
        if queue.full():  # never block producers; drop oldest for slow consumers
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(ev)

    # -- consumption --------------------------------------------------------------------------
    async def stream(self) -> AsyncIterator[Event]:
        """Yield events as they arrive until :meth:`close` is called."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Event | None] = asyncio.Queue(self._sub_max)
        entry = (loop, queue)
        with self._lock:
            self._subscribers.append(entry)
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    return
                yield ev
        finally:
            with self._lock:
                if entry in self._subscribers:
                    self._subscribers.remove(entry)

    @property
    def events(self) -> list[Event]:
        with self._lock:
            return list(self._events)

    def of_type(self, *types: EventType) -> list[Event]:
        wanted = set(types)
        return [e for e in self.events if e.type in wanted]

    # -- export -------------------------------------------------------------------------------
    def _persisted(self, ev: Event) -> dict[str, Any]:
        """Serializable form; with log_text=False every transcript/translation field is removed."""
        rec = ev.to_dict()
        if not self._log_text:
            rec["text"] = None
            rec["data"] = {k: (None if k in TEXT_DATA_KEYS else v) for k, v in rec["data"].items()}
        return rec

    def export_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for ev in self.events:
                rec = self._persisted(ev)
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["seq", "type", "t_ms", "wall_time", "segment_id", "text", "data"]
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for ev in self.events:
                rec = self._persisted(ev)
                rec["data"] = json.dumps(rec["data"], ensure_ascii=False, default=str)
                writer.writerow({c: rec.get(c) for c in cols})

    def close(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
            if self._jsonl is not None:
                self._jsonl.close()
                self._jsonl = None
        for loop, queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, None)


def stage_latencies(
    events: Iterable[Event], start: EventType, end: EventType
) -> list[float]:
    """Per-segment latency (ms) between the first ``start`` and first ``end`` event."""
    first_start: dict[int | None, float] = {}
    first_end: dict[int | None, float] = {}
    for ev in events:
        if ev.type == start:
            first_start.setdefault(ev.segment_id, ev.t_ms)
        elif ev.type == end:
            first_end.setdefault(ev.segment_id, ev.t_ms)
    return [first_end[k] - first_start[k] for k in first_start if k in first_end]
