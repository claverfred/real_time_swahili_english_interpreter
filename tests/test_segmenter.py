import numpy as np
import pytest

from src.config.schema import VadConfig
from src.pipeline.events import EventRecorder, EventType
from src.vad import VadSegmenter, segment_audio

SR, FS = 16000, 160   # 10 ms frames


class ScriptedVad:
    """Returns a scripted probability per frame, regardless of the audio."""
    name, sample_rate, frame_samples = "scripted", SR, FS

    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def __call__(self, frame):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset(self):
        self.i = 0


CFG = VadConfig(threshold=0.5, min_speech_ms=30, min_silence_ms=50, speech_pad_ms=20,
                hangover_ms=20, max_utterance_s=1.0)


def run(probs, cfg=CFG, chunk=FS * 4, recorder=None):
    audio = np.arange(len(probs) * FS, dtype=np.float32) / 1e6   # unique sample values
    seg = VadSegmenter(ScriptedVad(probs), cfg, recorder)
    events = []
    for i in range(0, len(audio), chunk):
        events += seg.process(audio[i:i + chunk])
    events += seg.flush()
    return audio, events, [e.segment for e in events if e.kind == "end"]


def test_single_utterance_with_pad_and_hangover():
    audio, _, segs = run([0] * 10 + [1] * 20 + [0] * 10)
    assert len(segs) == 1
    s = segs[0]
    assert s.start_s == pytest.approx(0.08)     # 2 pad frames before frame 10
    assert s.end_s == pytest.approx(0.32)       # last speech frame 29 + 2 hangover frames
    assert s.reason == "silence"
    assert np.array_equal(s.audio, audio[8 * FS:32 * FS])


def test_blip_shorter_than_min_speech_is_ignored():
    assert run([0] * 10 + [1, 1] + [0] * 10)[2] == []


def test_hysteresis_keeps_brief_dips_inside_speech():
    probs = [0] * 5 + [1] * 10 + [0.4] * 8 + [1] * 10 + [0] * 10
    assert len(run(probs)[2]) == 1


def test_gap_longer_than_min_silence_splits():
    probs = [0] * 5 + [1] * 10 + [0] * 8 + [1] * 10 + [0] * 10
    segs = run(probs)[2]
    assert [s.segment_id for s in segs] == [0, 1]


def test_max_utterance_split_is_contiguous():
    segs = run([1] * 250 + [0] * 10)[2]
    assert [s.reason for s in segs] == ["max_duration", "max_duration", "silence"]
    assert segs[0].duration_s == pytest.approx(1.0)
    assert segs[0].end_s == pytest.approx(segs[1].start_s)
    assert segs[1].end_s == pytest.approx(segs[2].start_s)


def test_flush_closes_open_segment():
    segs = run([0] * 5 + [1] * 30)[2]
    assert len(segs) == 1 and segs[0].reason == "flush"


@pytest.mark.parametrize("chunk", [37, 160, 1600, 100000])
def test_results_independent_of_chunk_size(chunk):
    probs = [0] * 7 + [1] * 13 + [0] * 9 + [1] * 30 + [0] * 3 + [1] * 5 + [0] * 12
    ref = [(s.start_s, s.end_s) for s in run(probs, chunk=FS)[2]]
    assert [(s.start_s, s.end_s) for s in run(probs, chunk=chunk)[2]] == ref


def test_streamed_audio_covers_segment_and_events_recorded():
    rec = EventRecorder()
    audio, events, segs = run([0] * 10 + [1] * 20 + [0] * 10, recorder=rec)
    streamed = np.concatenate([e.audio for e in events if e.kind in ("start", "audio")])
    # stream carries everything from segment start until the closing silence frame
    assert np.array_equal(streamed[: len(segs[0].audio)], segs[0].audio)
    kinds = [e.type for e in rec.events if e.type != EventType.AUDIO_FRAME_RECEIVED]
    assert kinds == [EventType.VAD_SPEECH_START, EventType.VAD_SPEECH_END]
    end = rec.of_type(EventType.VAD_SPEECH_END)[0]
    assert end.segment_id == 0 and end.data["reason"] == "silence"


def test_reset_restarts_ids_and_time():
    seg = VadSegmenter(ScriptedVad([0] * 3 + [1] * 10 + [0] * 10), CFG)
    x = np.zeros(23 * FS, dtype=np.float32)
    first = segment_audio(x, seg)
    seg.reset()
    second = segment_audio(x, seg)
    assert [(s.segment_id, s.start_s) for s in first] == [(s.segment_id, s.start_s) for s in second]
