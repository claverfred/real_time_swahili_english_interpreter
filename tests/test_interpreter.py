"""The async pipeline with fake models: ordering, flush, cancel, backpressure, failures."""

import asyncio
import time

import numpy as np
import pytest

from src.config import load_config
from src.mt import Translator
from src.pipeline import Components, EventType, RealtimeSwahiliInterpreter
from src.tts import Synthesizer
from tests.test_asr_streaming import FakeAsr
from tests.test_mt import FakeEngine
from tests.test_tts import FakeBackend

SR = 16000


class LevelVad:
    """Deterministic VAD: speech iff the frame RMS is above 0.05 (state-free, so reset is a no-op)."""
    name, sample_rate, frame_samples = "level", SR, 480

    def __call__(self, frame):
        return 1.0 if float(np.sqrt(np.mean(frame ** 2))) > 0.05 else 0.0

    def reset(self):
        pass


class SlowBackend(FakeBackend):
    def __init__(self, delay=0.15, **kw):
        super().__init__(**kw)
        self.delay = delay

    def synthesize(self, text):
        time.sleep(self.delay)
        return super().synthesize(text)


def speech(*parts):
    """parts: (seconds, is_speech) -> float32 audio; speech is a 0.3-amplitude tone."""
    out = []
    for secs, on in parts:
        t = np.arange(int(secs * SR)) / SR
        out.append((0.3 * np.sin(2 * np.pi * 220 * t) if on else np.zeros_like(t)).astype(np.float32))
    return np.concatenate(out)


TWO_UTTERANCES = speech((0.5, False), (2.0, True), (1.0, False), (1.5, True), (1.0, False))


def make(overrides=(), backend=None, asr=None, enable_tts=True, retry=None):
    cfg = load_config("cpu", ["vad.min_silence_ms=300", "vad.speech_pad_ms=0", "vad.hangover_ms=0",
                              "vad.min_speech_ms=60", *overrides], environ={})
    b = backend or FakeBackend()
    b.load()
    synth = Synthesizer([b], cfg.tts, retry_count=cfg.streaming.retry_count if retry is None else retry)
    comps = Components(vad=LevelVad(), asr=asr or FakeAsr(), translator=Translator(FakeEngine(), cfg.mt),
                       synthesizer=synth)
    return RealtimeSwahiliInterpreter(cfg, components=comps, enable_tts=enable_tts)


async def feed(interp, audio, block_s=0.1):
    step = int(block_s * SR)
    for i in range(0, len(audio), step):
        await interp.push_audio(audio[i:i + step], sample_rate=SR)


async def run_session(interp, audio, *, before_shutdown=None):
    await interp.start()
    events = []

    async def collect():
        async for ev in interp.events():
            events.append(ev)

    task = asyncio.create_task(collect())
    await feed(interp, audio)
    if before_shutdown:
        await before_shutdown(interp)
    summary = await interp.shutdown()
    await task
    return events, summary


def test_end_to_end_order_audio_and_latencies():
    interp = make()
    events, summary = asyncio.run(run_session(interp, TWO_UTTERANCES))
    types = [e.type for e in events]
    assert types.count(EventType.ASR_FINAL) == 2
    assert types.count(EventType.MT_FINAL) == 2
    assert types.count(EventType.PIPELINE_COMPLETE) == 2
    audio = [e for e in events if e.type == "tts_audio"]          # spec-style string comparison works
    assert audio and all(isinstance(e.audio, np.ndarray) and len(e.audio) for e in audio)
    for unit in (0, 1):  # per unit: asr_final -> mt_final -> first audio -> complete
        seq = {t: next(e.seq for e in events if e.type == t and e.data.get("unit_id", unit) == unit
                       and (t != EventType.ASR_FINAL or e.segment_id == unit))
               for t in (EventType.ASR_FINAL, EventType.MT_FINAL, EventType.TTS_FIRST_AUDIO,
                         EventType.PIPELINE_COMPLETE)}
        assert seq[EventType.ASR_FINAL] < seq[EventType.MT_FINAL] < seq[EventType.TTS_FIRST_AUDIO] \
            < seq[EventType.PIPELINE_COMPLETE]
    units = interp.units()
    assert [u["segment_ids"] for u in units] == [[0], [1]]
    assert units[0]["english"] == "today ninaenda market kununua" and units[0]["audio_s"] > 0
    lat = summary["latency_from_speech_end_ms"]
    assert lat["first_audio"] is not None and lat["mt_final"] is not None
    assert all(u["lat_asr_final_ms"] <= u["lat_mt_final_ms"] <= u["lat_first_audio_ms"]
               <= u["lat_full_audio_ms"] for u in units)
    assert summary["counters"]["units_complete"] == 2 and summary["errors"] == []
    assert interp.state == "stopped" and events[-1].data.get("status") == "stopped"


def test_flush_mid_utterance_finalizes_it():
    async def go():
        interp = make()
        await interp.start()
        await feed(interp, speech((0.3, False), (1.5, True)))      # still speaking: no VAD end yet
        assert await interp.flush(timeout=10) is True
        assert interp.counters["units_complete"] == 1
        assert interp.units()[0]["english"]
        st = interp.status()
        assert st["state"] == "running" and st["queues"]["audio"] == 0
        assert interp.health()["status"] == "ok"
        await interp.shutdown()
    asyncio.run(go())


def test_cancel_stops_speech_and_session_continues():
    async def go():
        interp = make(["tts.chunk_max_chars=20", "tts.first_chunk_max_chars=20"], backend=SlowBackend(0.2))
        await interp.start()
        stream = interp.events()
        await feed(interp, speech((0.3, False), (4.0, True), (0.6, False)))
        async for ev in stream:                                   # wait until it starts speaking
            if ev.type == EventType.TTS_FIRST_AUDIO:
                break
        await interp.cancel()
        await feed(interp, speech((0.3, False), (1.0, True), (0.6, False)))
        assert await interp.flush(timeout=10) is True
        await interp.shutdown()
        units = interp.units()
        assert units[0].get("cancelled") and "lat_full_audio_ms" not in units[0]
        assert units[-1]["english"] and not units[-1].get("cancelled")
        completes = [e for e in interp.recorder.events if e.type == EventType.TTS_COMPLETE]
        assert [e.data["unit_id"] for e in completes] == [units[-1]["unit_id"]]   # cancelled unit never completes
        assert interp.counters["cancels"] == 1
    asyncio.run(go())


def test_nonblocking_push_counts_overflow():
    async def go():
        interp = make(["streaming.queue_max=4"])
        await interp.start()
        chunk = np.zeros(1600, np.float32)
        results = [await interp.push_audio(chunk, block=False) for _ in range(10)]  # no await yields to workers
        assert results.count(False) == 6 and interp.counters["overflows"] == 6
        await interp.shutdown()
    asyncio.run(go())


def test_tts_failure_is_reported_and_text_still_delivered():
    interp = make(backend=FakeBackend(fail_times=10**6), retry=0)
    events, summary = asyncio.run(run_session(interp, TWO_UTTERANCES))
    types = [e.type for e in events]
    assert types.count(EventType.MT_FINAL) == 2 and types.count(EventType.PIPELINE_COMPLETE) == 2
    errors = [e for e in events if e.type == EventType.ERROR]
    assert len(errors) == 2 and errors[0].data["stage"] == "tts"
    assert all(u.get("tts_error") for u in interp.units())
    assert summary["counters"]["errors"] == 2


def test_asr_error_on_one_chunk_does_not_stop_session():
    class FlakyAsr(FakeAsr):
        def transcribe(self, audio, **kw):
            if not kw.get("fast") and not getattr(self, "failed", False):
                self.failed = True
                raise RuntimeError("decoder crashed")
            return super().transcribe(audio, **kw)

    interp = make(asr=FlakyAsr())
    events, _ = asyncio.run(run_session(interp, TWO_UTTERANCES))
    assert [e.data["stage"] for e in events if e.type == EventType.ERROR] == ["asr"]
    assert sum(e.type == EventType.MT_FINAL for e in events) == 1     # the second utterance still works
    assert interp.health()["state"] == "stopped"


def test_input_conversion_int16_stereo_and_resampling():
    async def go():
        interp = make(enable_tts=False)
        await interp.start()
        stereo = (np.ones((800, 2)) * 1000).astype(np.int16)        # 0.1 s at 8 kHz
        await interp.push_audio(stereo, sample_rate=8000)
        await interp.flush(timeout=10)
        assert interp.counters["samples_in"] == 1600
        summary = await interp.shutdown()
        assert summary["audio_in_s"] == 0.1
    asyncio.run(go())


def test_reset_clears_translation_context_and_text_only_mode():
    async def go():
        interp = make(enable_tts=False)
        await interp.start()
        await feed(interp, TWO_UTTERANCES)
        await interp.flush(timeout=10)
        assert interp._mt.history
        await interp.reset()
        assert interp._mt.history == [] and interp.counters["tts_chunks"] == 0
        assert interp.counters["units_complete"] == 2
        await interp.shutdown()
    asyncio.run(go())


def test_api_misuse_raises():
    interp = make()
    with pytest.raises(RuntimeError, match="start"):
        asyncio.run(interp.push_audio(np.zeros(10, np.float32)))
    assert interp.health()["status"] == "idle"


def test_model_status_reports_components():
    async def go():
        async with make() as interp:
            ms = interp.model_status()
            assert ms["asr"]["model"] == "fake" and ms["mt"]["model"] == "fake-mt"
            assert ms["tts"]["backend"] == "fake" and ms["vad"]["backend"] == "level"
            assert "first_inference_ms" in ms["tts"] and ms["runtime"]["device"] == "cpu"
        assert interp.state == "stopped"
    asyncio.run(go())
