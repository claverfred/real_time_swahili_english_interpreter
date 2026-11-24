"""Server: access control, live WebSocket turn-taking, busy lock, upload and downloads (fake models)."""

import io
import zipfile

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from src.config import load_config  # noqa: E402
from src.mt import Translator  # noqa: E402
from src.pipeline import Components  # noqa: E402
from src.server import AccessPolicy, create_app, is_loopback  # noqa: E402
from src.tts import Synthesizer  # noqa: E402
from tests.test_asr_streaming import FakeAsr  # noqa: E402
from tests.test_interpreter import SR, LevelVad, speech  # noqa: E402
from tests.test_mt import FakeEngine  # noqa: E402
from tests.test_tts import FakeBackend  # noqa: E402

UTT = speech((0.3, False), (1.5, True), (0.8, False))


def make_app(tmp_path, token=None, overrides=()):
    cfg = load_config("cpu", ["vad.min_silence_ms=300", "vad.speech_pad_ms=0", "vad.hangover_ms=0",
                              "vad.min_speech_ms=60", "server.consecutive_silence_ms=300",
                              f"storage.output_dir={tmp_path.as_posix()}/results", *overrides], environ={})
    b = FakeBackend()
    b.load()
    comps = Components(vad=LevelVad(), asr=FakeAsr(), translator=Translator(FakeEngine(), cfg.mt),
                       synthesizer=Synthesizer([b], cfg.tts))
    policy = AccessPolicy(token, required=bool(token))
    return create_app(cfg, policy, components=comps)


def recv_until(ws, pred, limit=500):
    """Receive messages until pred(msg) is true; returns all JSON messages seen (binary counted)."""
    import json
    seen = []
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            seen.append({"type": "_binary", "n": len(m["bytes"])})
            continue
        if m.get("text") is None:
            raise AssertionError(f"socket closed: {m}")
        msg = json.loads(m["text"])
        seen.append(msg)
        if pred(msg):
            return seen
    raise AssertionError("expected message not received")


def send_audio(ws, audio, block_s=0.1):
    step = int(block_s * SR)
    for i in range(0, len(audio), step):
        ws.send_bytes(audio[i:i + step].astype("<f4").tobytes())


# -- security --------------------------------------------------------------------------------------
def test_policy_rules():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5")
    pol, generated = AccessPolicy.for_host("0.0.0.0", None)
    assert pol.required and generated and len(pol.token) > 20
    assert not pol.token_ok(None) and not pol.token_ok("wrong") and pol.token_ok(pol.token)
    local, gen2 = AccessPolicy.for_host("127.0.0.1", None)
    assert not local.required and not gen2 and local.token_ok(None)
    assert local.origin_ok(None, "127.0.0.1:8000")
    assert local.origin_ok("http://127.0.0.1:8000", "127.0.0.1:8000")
    assert not local.origin_ok("http://evil.example", "127.0.0.1:8000")
    rl = AccessPolicy(None, required=False, rate_limit_per_min=3)
    assert [rl.rate_ok("a") for _ in range(4)] == [True, True, True, False]
    assert rl.rate_ok("b")


def test_api_token_origin_and_index(tmp_path):
    with TestClient(make_app(tmp_path, token="s3cret")) as c:
        assert c.get("/").status_code == 200 and "interpreter" in c.get("/").text
        assert c.get("/static/app.js").status_code == 200
        assert c.get("/api/status").status_code == 401
        assert c.get("/api/status?token=wrong").status_code == 401
        r = c.get("/api/status", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200 and r.json()["pool"]["state"] == "ready"
        assert c.get("/api/health?token=s3cret").json()["status"] == "ok"
        r = c.get("/api/status?token=s3cret", headers={"Origin": "http://evil.example"})
        assert r.status_code == 403
        assert r.headers["x-content-type-options"] == "nosniff"


def test_ws_rejects_bad_token_and_cross_origin(tmp_path):
    with TestClient(make_app(tmp_path, token="s3cret")) as c:
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/ws/live") as ws:
                ws.receive_text()
        assert e.value.code == 4401
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/ws/live?token=s3cret", headers={"Origin": "http://evil.example"}) as ws:
                ws.receive_text()
        assert e.value.code == 4403


def test_ws_bad_start_message(tmp_path):
    with TestClient(make_app(tmp_path)) as c, c.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "start", "sample_rate": 12})
        msg = ws.receive_json()
        assert msg["type"] == "error" and msg["code"] == "bad_request"


# -- live session ----------------------------------------------------------------------------------
def test_ws_consecutive_turn_taking_and_export(tmp_path):
    with TestClient(make_app(tmp_path)) as c:
        with c.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "start", "sample_rate": SR, "mode": "consecutive", "tts": True})
            ready = recv_until(ws, lambda m: m["type"] == "ready")[-1]
            assert ready["mode"] == "consecutive" and ready["models"]["asr"] == "fake"
            # first sentence -> the server closes the turn, interprets, and reports it spoken
            send_audio(ws, UTT)
            seen = recv_until(ws, lambda m: m["type"] == "turn" and m["state"] == "spoken")
            types = [m["type"] for m in seen]
            assert {"type": "turn", "state": "interpreting"} in seen
            evs = [m["event"]["type"] for m in seen if m["type"] == "event"]
            assert "asr_final" in evs and "mt_final" in evs and "pipeline_complete" in evs
            i = types.index("tts_audio")
            assert seen[i + 1]["type"] == "_binary" and seen[i + 1]["n"] == 2 * seen[i]["samples"]
            # audio while the turn is closed is ignored (replaced by silence)
            send_audio(ws, UTT)
            ws.send_json({"type": "listening"})
            send_audio(ws, UTT)            # second turn
            recv_until(ws, lambda m: m["type"] == "turn" and m["state"] == "spoken")
            ws.send_json({"type": "stop"})
            summary = recv_until(ws, lambda m: m["type"] == "summary")[-1]
        assert summary["summary"]["units"] == 2              # the muted sentence was not interpreted
        assert summary["summary"]["capture"]["turn_muted_s"] > 1.0
        run_id = summary["run_id"]
        assert "speech_en.wav" in summary["files"] and "manifest.json" in summary["files"]
        z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/runs/{run_id}/export.zip").content))
        assert {n.split("/")[1] for n in z.namelist()} >= {"translation_en.txt", "units.jsonl", "speech_en.wav"}
        assert c.get(f"/api/runs/{run_id}/speech_en.wav").status_code == 200
        assert c.get(f"/api/runs/{run_id}/config.yaml").status_code == 404
        assert c.get("/api/runs/..%2F..%2Fsecrets/manifest.json").status_code == 404
        assert c.get("/api/runs/nope/manifest.json").status_code == 404


def test_ws_simultaneous_cancel_and_busy(tmp_path):
    with TestClient(make_app(tmp_path)) as c:
        with c.websocket_connect("/ws/live") as ws:
            ws.send_json({"type": "start", "sample_rate": 48000, "mode": "simultaneous", "export": False})
            recv_until(ws, lambda m: m["type"] == "ready")
            # a second client is refused while this session holds the models
            with c.websocket_connect("/ws/live") as ws2:
                ws2.send_json({"type": "start", "sample_rate": SR})
                msg = ws2.receive_json()
                assert msg["type"] == "error" and msg["code"] == "busy"
            assert c.get("/api/health").json()["busy"] is True
            up = np.repeat(UTT, 3)                       # 48 kHz client audio, resampled by the server
            send_audio(ws, up, block_s=0.1 / 3)
            ws.send_json({"type": "cancel"})
            recv_until(ws, lambda m: m["type"] == "turn" and m["state"] == "cancelled")
            ws.send_bytes(b"\x00\x01\x02")             # not float32 -> error, session continues
            recv_until(ws, lambda m: m["type"] == "error" and m["code"] == "bad_audio")
            ws.send_json({"type": "ping"})
            recv_until(ws, lambda m: m["type"] == "pong")
            ws.send_json({"type": "stop"})
            summary = recv_until(ws, lambda m: m["type"] == "summary")[-1]
            assert summary["run_id"] is None
        assert c.get("/api/health").json()["busy"] is False


# -- upload ----------------------------------------------------------------------------------------
def _wav_bytes(tmp_path, audio):
    from src.audio import save_wav
    p = save_wav(tmp_path / "in.wav", audio, SR)
    return p.read_bytes()


def test_upload_interprets_and_cleans_up(tmp_path, monkeypatch):
    import tempfile
    made = []
    real = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kw: made.append(real(dir=tmp_path, **kw)) or made[-1])
    audio = np.concatenate([UTT, UTT])
    with TestClient(make_app(tmp_path)) as c:
        r = c.post("/api/interpret", files={"file": ("talk.wav", _wav_bytes(tmp_path, audio), "audio/wav")},
                   data={"tts": "true"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["units"]) == 2 and body["english"]
        assert "speech_en.wav" in body["files"]
        assert c.get(f"/api/runs/{body['run_id']}/translation_en.txt").text.strip()
    assert made and not any(__import__("os").path.exists(d) for d in made)   # upload deleted


def test_upload_limits(tmp_path):
    with TestClient(make_app(tmp_path, overrides=["audio.max_input_mb=0.01"])) as c:
        big = _wav_bytes(tmp_path, np.concatenate([UTT] * 3))
        r = c.post("/api/interpret", files={"file": ("big.wav", big, "audio/wav")})
        assert r.status_code == 413
        r = c.post("/api/interpret", files={"file": ("x.wav", b"not audio", "audio/wav")})
        assert r.status_code == 400
