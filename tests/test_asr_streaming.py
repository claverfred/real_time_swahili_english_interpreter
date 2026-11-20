import numpy as np

from src.asr import AsrResult, StreamingTranscriber
from src.config.schema import AsrConfig, VadConfig
from src.pipeline.events import EventRecorder, EventType
from src.vad import VadSegmenter

SR = 16000
WORDS = "leo ninaenda sokoni kununua matunda na mboga kwa familia yangu yote".split()


class FakeAsr:
    """One word per 0.5 s of audio. Optional scripted partial texts to exercise agreement."""
    model_name = "fake"

    def __init__(self, partial_script=None, ms=10.0):
        self.partial_script = list(partial_script or [])
        self.calls = []
        self.ms = ms

    def transcribe(self, audio, *, fast=False, initial_prompt=None, previous_text="", word_timestamps=None):
        self.calls.append(("partial" if fast else "final", len(audio) / SR, initial_prompt))
        if fast and self.partial_script:
            text = self.partial_script.pop(0)
        else:
            text = " ".join(WORDS[: int(len(audio) / SR / 0.5)])
        return AsrResult(text, "sw", True, len(audio) / SR, self.ms, "fake")


class ScriptedVad:
    name, sample_rate, frame_samples = "scripted", SR, 160

    def __init__(self, probs):
        self.probs, self.i = probs, 0

    def __call__(self, frame):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset(self):
        self.i = 0


def drive(asr, speech_frames=300, cfg=None, vad_cfg=None, recorder=None, chunk=1600):
    vad_cfg = vad_cfg or VadConfig(min_speech_ms=30, min_silence_ms=300, speech_pad_ms=0, hangover_ms=0)
    cfg = cfg or AsrConfig(model="fake", chunk_s=1.0, overlap_s=0.5, stable_prefix_agreement=2)
    probs = [0] * 20 + [1] * speech_frames + [0] * 60
    seg = VadSegmenter(ScriptedVad(probs), vad_cfg)
    st = StreamingTranscriber(asr, cfg, SR, recorder, partials=True)
    audio = np.zeros(len(probs) * 160, np.float32)
    ups = []
    for i in range(0, len(audio), chunk):
        for ev in seg.process(audio[i:i + chunk]):
            ups += st.on_vad_event(ev)
    for ev in seg.flush():
        ups += st.on_vad_event(ev)
    return st, ups


def test_partials_then_one_final_per_segment():
    st, ups = drive(FakeAsr())
    kinds = [u.kind for u in ups]
    assert kinds.count("final") == 1 and kinds[-1] == "final"
    assert kinds.count("partial") >= 2
    final = ups[-1]
    assert final.text.split()[:5] == WORDS[:5]
    assert final.start_s == 0.2 and final.end_s > 3.0


def test_committed_prefix_is_monotonic_and_needs_agreement():
    script = ["leo nina", "leo ninaenda soko", "leo ninaenda sokoni ku", "leo ninaenda sokoni kununua"]
    st, ups = drive(FakeAsr(partial_script=script))
    partials = [u for u in ups if u.kind == "partial"]
    assert partials[0].stable == ""                     # one hypothesis: nothing agreed yet
    assert partials[1].stable == "leo"                  # agreement of hyps 1-2
    assert partials[2].stable == "leo ninaenda"
    stables = [len(p.stable.split()) for p in partials]
    assert stables == sorted(stables)


def test_disagreement_counts_revision_without_shrinking():
    script = ["leo ninaenda", "leo ninaenda", "jana tulienda", "jana tulienda"]
    st, ups = drive(FakeAsr(partial_script=script))
    partials = [u for u in ups if u.kind == "partial"]
    assert partials[1].stable == "leo ninaenda"
    assert partials[2].stable == "leo ninaenda"          # never rewritten silently
    assert partials[2].revisions >= 1


def test_adaptive_interval_reduces_partials_on_slow_hardware():
    fast = drive(FakeAsr(ms=10))[1]
    slow = drive(FakeAsr(ms=1500))[1]                   # 1.5 s per decode -> wait >= 2.25 s
    n = lambda ups: sum(u.kind == "partial" for u in ups)  # noqa: E731
    assert n(slow) < n(fast)


def test_max_duration_split_carries_overlap_and_dedupes():
    vad_cfg = VadConfig(min_speech_ms=30, min_silence_ms=300, speech_pad_ms=0, hangover_ms=0, max_utterance_s=2.0)
    asr = FakeAsr()
    st, ups = drive(asr, speech_frames=400, vad_cfg=vad_cfg)
    finals = [u for u in ups if u.kind == "final"]
    assert [f.extra["vad_reason"] for f in finals][:2] == ["max_duration", "max_duration"]
    final_calls = [c for c in asr.calls if c[0] == "final"]
    assert final_calls[0][1] == 2.0 and final_calls[1][1] == 2.5     # 0.5 s overlap prepended
    # second segment's text starts after the words already emitted
    assert finals[1].text.split()[0] != finals[0].text.split()[0]


def test_events_recorded_with_latency():
    rec = EventRecorder()
    drive(FakeAsr(), recorder=rec)
    finals = rec.of_type(EventType.ASR_FINAL)
    assert len(finals) == 1 and finals[0].data["latency_ms"] == 10.0
    assert rec.of_type(EventType.ASR_PARTIAL)


def test_prompt_only_when_conditioning_enabled():
    asr = FakeAsr()
    drive(asr)
    assert all(c[2] is None for c in asr.calls)
    asr2 = FakeAsr()
    cfg = AsrConfig(model="fake", chunk_s=1.0, overlap_s=0.5, condition_on_previous_text=True)
    vad_cfg = VadConfig(min_speech_ms=30, min_silence_ms=300, speech_pad_ms=0, hangover_ms=0, max_utterance_s=2.0)
    drive(asr2, speech_frames=400, cfg=cfg, vad_cfg=vad_cfg)
    assert any(c[2] for c in asr2.calls)
