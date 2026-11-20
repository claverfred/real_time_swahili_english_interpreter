import asyncio
import csv
import json

import pytest

from src.pipeline.events import EventRecorder, EventType, stage_latencies
from src.utils.stats import percentile, summarize


def test_events_are_ordered_and_timestamped():
    rec = EventRecorder(session_id="s1")
    a = rec.emit(EventType.VAD_SPEECH_START, segment_id=0)
    b = rec.emit("asr_final", segment_id=0, text="Habari za asubuhi", confidence=0.9)
    assert (a.seq, b.seq) == (0, 1)
    assert b.t_ms >= a.t_ms
    assert b.data == {"confidence": 0.9}
    assert [e.type for e in rec.events] == [EventType.VAD_SPEECH_START, EventType.ASR_FINAL]


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError):
        EventRecorder().emit("not_an_event")


def test_jsonl_and_csv_export_keep_unicode_and_drop_audio(tmp_path):
    rec = EventRecorder(jsonl_path=tmp_path / "live.jsonl")
    rec.emit(EventType.ASR_FINAL, segment_id=1, text="Ninaenda sokoni — leo.")
    rec.emit(EventType.TTS_AUDIO, segment_id=1, audio=b"\x00\x01")
    rec.export_jsonl(tmp_path / "events.jsonl")
    rec.export_csv(tmp_path / "events.csv")
    rec.close()
    lines = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["text"] == "Ninaenda sokoni — leo."
    assert "audio" not in lines[1]
    live = (tmp_path / "live.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(live) == 2
    rows = list(csv.DictReader((tmp_path / "events.csv").open(encoding="utf-8")))
    assert rows[0]["type"] == "asr_final"


def test_log_text_false_redacts_persisted_text(tmp_path):
    rec = EventRecorder(log_text=False)
    rec.emit(EventType.ASR_FINAL, text="siri")
    rec.export_jsonl(tmp_path / "e.jsonl")
    assert json.loads((tmp_path / "e.jsonl").read_text())["text"] is None
    assert rec.events[0].text == "siri"  # live consumers still get it


def test_async_stream_receives_events_until_close():
    async def scenario():
        rec = EventRecorder()
        got = []

        async def consume():
            async for ev in rec.stream():
                got.append(ev.type)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        rec.emit(EventType.ASR_PARTIAL, text="hab")
        rec.emit(EventType.ASR_FINAL, text="habari")
        await asyncio.sleep(0.01)
        rec.close()
        await asyncio.wait_for(task, 1)
        return got

    assert asyncio.run(scenario()) == [EventType.ASR_PARTIAL, EventType.ASR_FINAL]


def test_stage_latencies_per_segment():
    rec = EventRecorder()
    for seg in (0, 1):
        rec.emit(EventType.VAD_SPEECH_END, segment_id=seg)
        rec.emit(EventType.TTS_FIRST_AUDIO, segment_id=seg)
    lat = stage_latencies(rec.events, EventType.VAD_SPEECH_END, EventType.TTS_FIRST_AUDIO)
    assert len(lat) == 2 and all(x >= 0 for x in lat)


def test_percentiles_match_numpy_linear():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 50) == 5.5
    assert percentile(xs, 95) == pytest.approx(9.55)
    s = summarize(xs)
    assert s["n"] == 10 and s["p99"] == pytest.approx(9.91) and s["mean"] == 5.5
    assert summarize([])["n"] == 0
