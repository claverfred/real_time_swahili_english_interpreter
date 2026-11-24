"""FastAPI app: browser UI, live WebSocket interpretation, file upload, results download.

    GET  /                          browser UI (static; no data)
    GET  /api/health                server and model state
    GET  /api/status                models, limits, active session
    WS   /ws/live                   live interpretation (protocol in :mod:`src.server.live`)
    POST /api/interpret             upload an audio file -> Swahili, English, latencies, run id
    GET  /api/runs/{run_id}/{file}  download a result of this server's sessions (or export.zip)

Every /api and /ws request passes :class:`~src.server.security.AccessPolicy` (token, origin,
rate limit). Only one session uses the models at a time; others get HTTP 409 / close 4409.
Uploaded audio is written to a temporary file that is deleted when the request ends;
microphone audio is never written to disk.
"""

from __future__ import annotations

import asyncio
import io
import re
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config.schema import InterpreterConfig
from ..pipeline import Components, EventType, RealtimeSwahiliInterpreter
from .live import CLOSE_FORBIDDEN, CLOSE_UNAUTHORIZED, LiveSession
from .pool import ModelPool, PoolBusy
from .security import AccessPolicy, presented_token

STATIC = Path(__file__).parent / "static"
RUN_FILES = {"transcript_sw.txt", "translation_en.txt", "units.jsonl", "speech_en.wav", "events.jsonl",
             "events.csv", "manifest.json"}
_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_SUFFIX = re.compile(r"\.[a-z0-9]{1,6}")
CONSENT = ("Your microphone audio is sent only to this interpreter server and processed there. It is not "
           "saved. Make sure everyone being recorded has agreed to it.")


def create_app(cfg: InterpreterConfig, policy: AccessPolicy, *, components: Components | None = None,
               local_files_only: bool = False, user_set_silence: bool = False) -> FastAPI:
    from ..platform.storage import prepare_storage

    paths = prepare_storage(cfg)
    output_dir = Path(paths["output_dir"])
    pool = ModelPool(cfg, components=components, local_files_only=local_files_only)
    runs: dict[str, Path] = {}
    sessions: set[LiveSession] = set()

    def active_interp() -> RealtimeSwahiliInterpreter | None:
        """The interpreter of the live session that holds the models (rejected ones have none)."""
        return next((s.interp for s in sessions if s.interp is not None), None)

    def register_run(run_dir: Path) -> str:
        runs[run_dir.name] = run_dir
        return run_dir.name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.server.preload:
            pool.start_loading()
        yield

    app = FastAPI(title="Swahili-English interpreter", version="0.7", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.pool, app.state.runs, app.state.policy = pool, runs, policy

    def guard(request: Request) -> None:
        client = request.client.host if request.client else "?"
        if not policy.rate_ok(client):
            raise HTTPException(429, "too many requests")
        if not policy.origin_ok(request.headers.get("origin"), request.headers.get("host")):
            raise HTTPException(403, "cross-origin request refused")
        if not policy.token_ok(presented_token(request.headers, request.query_params)):
            raise HTTPException(401, "missing or wrong token", headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp: Response | None = None
        if request.url.path.startswith("/api/"):
            try:
                guard(request)
            except HTTPException as exc:
                resp = JSONResponse({"detail": exc.detail}, exc.status_code, headers=exc.headers)
        if resp is None:
            resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Cache-Control", "no-store")
        resp.headers.setdefault("Content-Security-Policy",
                                "default-src 'self'; connect-src 'self' ws: wss:; media-src 'self' blob:; "
                                "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return resp

    # -- UI ----------------------------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # -- API ---------------------------------------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        it = active_interp()
        return {"status": {"ready": "ok", "loading": "loading", "failed": "failed"}.get(pool.state, pool.state),
                "models": pool.state, "busy": pool.busy, "session": it.health() if it else None}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        it = active_interp()
        return {"profile": cfg.profile, "pool": pool.status(),
                "asr": {"model": cfg.asr.model, "beam_size": cfg.asr.beam_size,
                        "corrections": bool(cfg.asr.corrections_path)},
                "mt": {"model": cfg.mt.model, "license": cfg.mt.license},
                "tts": {"backend": cfg.tts.backend, "voice": cfg.tts.voice},
                "limits": {"max_upload_mb": cfg.audio.max_input_mb, "max_upload_s": cfg.audio.max_input_seconds,
                           "max_session_s": cfg.server.max_session_s, "idle_timeout_s": cfg.server.idle_timeout_s},
                "token_required": policy.required, "consent": CONSENT, "log_text": cfg.logging.log_text,
                "session": it.status() if it else None}

    @app.post("/api/interpret")
    async def interpret_file(file: Annotated[UploadFile, File()],
                             tts: Annotated[bool, Form()] = True) -> dict[str, Any]:
        from ..audio import AudioError

        max_bytes = int(cfg.audio.max_input_mb * 2**20)
        tmp_dir = Path(tempfile.mkdtemp(prefix="swi-upload-"))
        suffix = Path(file.filename or "audio.wav").suffix.lower()[:8]
        if not _SUFFIX.fullmatch(suffix):
            suffix = ".wav"
        tmp = tmp_dir / ("upload" + suffix)
        try:
            size = 0
            with tmp.open("wb") as fh:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(413, f"file is larger than {cfg.audio.max_input_mb} MB")
                    fh.write(chunk)
            try:
                async with pool.lease("file upload") as comps:
                    return await _interpret_upload(tmp, file.filename or "upload", tts, comps)
            except PoolBusy as exc:
                raise HTTPException(409, str(exc)) from exc
            except AudioError as exc:
                raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)    # uploaded audio never outlives the request

    async def _interpret_upload(tmp: Path, name: str, tts: bool, comps: Components) -> dict[str, Any]:
        from ..audio import load_audio
        from ..pipeline.export import export_session

        interp = RealtimeSwahiliInterpreter(cfg, components=comps, enable_tts=tts, partials=False,
                                            local_files_only=True)
        sr = interp.sample_rate
        audio, info = await asyncio.get_running_loop().run_in_executor(
            None, lambda: load_audio(tmp, sr, cfg.audio.max_input_seconds, cfg.audio.max_input_mb, interp.fallbacks))
        await interp.start()
        speech: list[np.ndarray] = []
        pause = np.zeros(int(cfg.tts.sample_rate * cfg.tts.sentence_pause_ms / 1000), np.float32)

        async def collect() -> None:
            async for ev in interp.events():
                if ev.type == EventType.TTS_AUDIO and ev.audio is not None:
                    speech.extend([np.asarray(ev.audio, np.float32), pause])

        task = asyncio.create_task(collect())
        step = int(0.5 * sr)
        for i in range(0, len(audio), step):
            await interp.push_audio(audio[i:i + step], sr)
        summary = await interp.shutdown(drain=True)
        await task
        summary["pacing"] = "batch (latencies include backlog)"
        wav = np.concatenate(speech[:-1]) if speech else None
        input_info = {**info.to_dict(), "path": None, "name": Path(name).name}
        run_dir = await asyncio.get_running_loop().run_in_executor(None, lambda: export_session(
            interp, output_dir, mode="interpret-upload", speech=wav, speech_sr=cfg.tts.sample_rate,
            summary=summary, extra={"input": input_info}))
        run_id = register_run(run_dir)
        units = interp.units()
        return {"run_id": run_id, "units": units, "summary": summary,
                "swahili": " ".join(u["swahili"] for u in units if u["swahili"]).strip(),
                "english": " ".join(u["english"] for u in units if u["english"]).strip(),
                "files": sorted(p.name for p in run_dir.iterdir()), "state": interp.state}

    @app.get("/api/runs/{run_id}/{name}")
    async def run_file(run_id: str, name: str) -> Response:
        if not _RUN_ID.match(run_id) or run_id not in runs:
            raise HTTPException(404, "unknown run")
        run_dir = runs[run_id]
        if name == "export.zip":
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(run_dir.iterdir()):
                    if p.is_file() and p.name in RUN_FILES:
                        zf.write(p, f"{run_id}/{p.name}")
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})
        if name not in RUN_FILES or not (run_dir / name).is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(run_dir / name, filename=name)

    # -- live WebSocket ----------------------------------------------------------------------------
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        if not policy.origin_ok(ws.headers.get("origin"), ws.headers.get("host")):
            await ws.close(CLOSE_FORBIDDEN, "cross-origin connection refused")
            return
        if not policy.token_ok(presented_token(ws.headers, ws.query_params)):
            await ws.close(CLOSE_UNAUTHORIZED, "missing or wrong token")
            return
        session = LiveSession(ws, pool, cfg, output_dir=output_dir, register_run=register_run,
                              user_set_silence=user_set_silence)
        sessions.add(session)
        try:
            await session.run()
        finally:
            sessions.discard(session)

    return app
