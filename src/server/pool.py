"""Models loaded once per server process and lent to one session at a time.

Loading the cpu profile takes 10-20 s, so the server loads VAD, ASR, MT and TTS once (in the
background at startup with ``server.preload``) and gives each session the same objects through
:class:`~src.pipeline.Components`. The models are not safe to use from two sessions at once, so
:meth:`ModelPool.lease` hands them out exclusively; a second client gets "busy" instead of
waiting behind a session that may last half an hour.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from ..config.schema import InterpreterConfig
from ..pipeline import Components

PoolState = Literal["idle", "loading", "ready", "failed"]


class PoolBusy(RuntimeError):
    pass


class ModelPool:
    def __init__(self, cfg: InterpreterConfig, *, components: Components | None = None,
                 local_files_only: bool = False) -> None:
        self.cfg = cfg
        self.local_files_only = local_files_only
        self.components: Components | None = components
        self.state: PoolState = "ready" if components is not None else "idle"
        self.error: str | None = None
        self.load_ms: float | None = None
        self.info: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._load_task: asyncio.Task[None] | None = None
        self.holder: str | None = None          # what currently holds the models

    # -- loading ---------------------------------------------------------------------------------
    def start_loading(self) -> asyncio.Task[None]:
        if self._load_task is None:
            self._load_task = asyncio.create_task(self._load())
        return self._load_task

    async def ready(self) -> Components:
        if self.state != "ready":
            await self.start_loading()
        if self.state != "ready" or self.components is None:
            raise RuntimeError(f"models failed to load: {self.error}")
        return self.components

    async def _load(self) -> None:
        if self.components is not None:
            self.state = "ready"
            return
        self.state = "loading"
        t0 = time.perf_counter()
        try:
            self.components = await asyncio.get_running_loop().run_in_executor(None, self._build)
            self.state = "ready"
        except Exception as exc:
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
        self.load_ms = round((time.perf_counter() - t0) * 1000, 1)

    def _build(self) -> Components:
        from ..asr import create_asr
        from ..mt import create_translator
        from ..platform.resolve import resolve_runtime
        from ..platform.storage import prepare_storage
        from ..tts import create_synthesizer
        from ..vad import create_vad

        cfg, runtime, fallbacks = resolve_runtime(self.cfg)
        paths = prepare_storage(cfg)
        vad = create_vad(cfg.vad, cfg.audio.sample_rate, cfg.audio.frame_ms, str(paths["cache_dir"]), fallbacks,
                         allow_fallback=cfg.fallback.policy != "fail")
        asr = create_asr(cfg, fallbacks, local_files_only=self.local_files_only)
        tr = create_translator(cfg, fallbacks, local_files_only=self.local_files_only)
        syn = create_synthesizer(cfg, fallbacks)
        self.info = {"vad": vad.name, "asr": asr.model_name, "mt": tr.model_name, "tts": syn.backend.name,
                     "device": runtime.device, "fallbacks": fallbacks.to_list()}
        return Components(vad=vad, asr=asr, translator=tr, synthesizer=syn)

    # -- leasing ---------------------------------------------------------------------------------
    @asynccontextmanager
    async def lease(self, holder: str):
        """Exclusive use of the loaded models; raises :class:`PoolBusy` if a session has them."""
        if self._lock.locked():
            raise PoolBusy(f"the interpreter is in use ({self.holder})")
        async with self._lock:
            self.holder = holder
            try:
                yield await self.ready()
            finally:
                self.holder = None

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "error": self.error, "load_ms": self.load_ms, "busy": self.busy,
                "holder": self.holder, "models": self.info}
