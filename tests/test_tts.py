import numpy as np
import pytest

from src.config import load_config
from src.config.schema import MtConfig, TtsConfig
from src.mt import MtStage, Translator
from src.pipeline.events import EventRecorder, EventType
from src.tts import (
    PronunciationDict,
    Synthesizer,
    TtsError,
    TtsStage,
    chunk_text,
    normalize_for_tts,
)
from src.tts.audio import apply_fades, normalize_rms, postprocess, trim_silence, validate_output
from src.tts.backends import TtsBackend, piper_voice_path
from src.tts.factory import create_synthesizer

CFG = TtsConfig(model="fake", voice="v")


class FakeBackend(TtsBackend):
    """A 200 Hz tone, 60 ms per character, padded with silence; can be told to fail."""

    def __init__(self, name="fake", sr=16000, fail_times=0, fail_load=False):
        super().__init__("v")
        self.name, self.sr, self.fail_times, self.fail_load = name, sr, fail_times, fail_load
        self.calls = []

    def load(self):
        if self.fail_load:
            raise TtsError("cannot load")
        self.sample_rate = self.sr
        self.load_info = {"backend": self.name}
        return self.load_info

    def synthesize(self, text):
        self.calls.append(text)
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("network down")
        n = int(self.sr * 0.06 * len(text))
        t = np.arange(n) / self.sr
        tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        pad = np.zeros(int(self.sr * 0.3), np.float32)
        return np.concatenate([pad, tone, pad])


def test_chunking_is_sentence_safe_with_short_first_chunk():
    text = ("On Monday, the President of Tanzania spoke to citizens in Dodoma, the capital city, about the "
            "new budget. Dr. Mwinyi agreed. The price is 2,500 shillings.")
    assert chunk_text(text, max_chars=120, first_max_chars=40)[0].endswith("budget.")  # no clause in 20..40
    chunks = chunk_text(text, max_chars=120, first_max_chars=70)
    assert chunks[0] == "On Monday, the President of Tanzania spoke to citizens in Dodoma,"
    assert chunks[1] == "the capital city, about the new budget."
    assert chunks[2:] == ["Dr. Mwinyi agreed.", "The price is 2,500 shillings."]   # not split at "Dr."
    assert all(len(c) <= 120 for c in chunks)


def test_chunking_long_sentence_without_clauses_breaks_at_words():
    long = " ".join(["word"] * 60) + "."
    chunks = chunk_text(long, max_chars=50, first_max_chars=None)
    assert all(len(c) <= 50 for c in chunks) and " ".join(chunks) == long
    assert chunk_text("") == []


def test_normalize_for_tts():
    assert normalize_for_tts("CCM won 60% & TANESCO agreed.") == "C C M won 60 percent and TANESCO agreed."
    assert normalize_for_tts("It costs TSh 2,000.") == "It costs 2,000 Tanzanian shillings."
    pron = PronunciationDict({"Dodoma": "Doh-DOH-mah", "CCM": "Chama Cha Mapinduzi"})
    assert normalize_for_tts("CCM met in dodoma.", pron=pron) == "Chama Cha Mapinduzi met in Doh-DOH-mah."
    assert normalize_for_tts("UN", spell_acronyms=False) == "UN"


def test_pronunciation_dict_load(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("word,spoken\nNyerere,Nyeh-REH-reh\n", encoding="utf-8")
    assert PronunciationDict.load(p).apply("Julius Nyerere") == "Julius Nyeh-REH-reh"


def test_audio_postprocessing():
    sr = 16000
    audio = FakeBackend().synthesize("hello")
    trimmed = trim_silence(audio, sr)
    assert len(trimmed) < len(audio) - sr * 0.5
    norm = normalize_rms(trimmed, -20.0)
    rms_db = 20 * np.log10(np.sqrt(np.mean(norm ** 2)))
    assert abs(rms_db + 20) < 0.5 and np.max(np.abs(norm)) <= 10 ** (-1 / 20) + 1e-6
    faded = apply_fades(np.ones(1000, np.float32), sr, 5)
    assert faded[0] == 0 and faded[-1] == 0 and faded[500] == 1
    out = postprocess(audio, sr, 22050, trim=True, fade_ms=5, target_rms_dbfs=-20)
    assert abs(len(out) / 22050 - len(trimmed) / sr) < 0.01
    assert validate_output(np.zeros(0, np.float32), sr, "x") == ["empty_audio"]
    assert "mostly_silent" in validate_output(np.zeros(sr * 2, np.float32) + 1e-5, sr, "a reasonably long text here")


def test_synthesizer_output_rate_pause_and_first_audio():
    syn = Synthesizer([FakeBackend()], CFG)
    syn.backend.load()
    res = syn.synthesize("Good morning. How are you today?")
    assert res.sample_rate == 22050 and len(res.chunks) == 2
    assert res.first_audio_ms <= res.total_ms
    pause = int(22050 * 0.12)
    assert len(res.audio) == sum(len(c.audio) for c in res.chunks) + pause
    assert all(c.warnings == [] for c in res.chunks)


def test_runtime_fallback_after_retries_is_recorded():
    primary, backup = FakeBackend("edge", fail_times=10), FakeBackend("piper")
    primary.load()
    seen = []
    syn = Synthesizer([primary, backup], CFG, retry_count=2, on_fallback=lambda *a: seen.append(a))
    chunk = syn.synthesize_chunk("Hello there.")
    assert chunk.backend == "piper" and len(primary.calls) == 3
    assert seen[0][:2] == ("edge", "piper") and syn.switches[0]["to"] == "piper"
    assert syn.synthesize_chunk("Again.").backend == "piper"      # stays switched


def test_retry_recovers_without_fallback():
    b = FakeBackend(fail_times=1)
    b.load()
    syn = Synthesizer([b, FakeBackend("other")], CFG, retry_count=1)
    assert syn.synthesize_chunk("Hi there.").backend == "fake" and syn.switches == []


def test_all_backends_failing_raises():
    b = FakeBackend(fail_times=99)
    b.load()
    with pytest.raises(TtsError):
        Synthesizer([b], CFG, retry_count=0).synthesize_chunk("Hi.")


def test_factory_startup_fallback_logged():
    from src.utils.manifest import FallbackLog
    cfg = load_config("cpu", ["tts.fallback_backends=[sapi]"])
    fb = FallbackLog()
    syn = create_synthesizer(cfg, fb, backends=[FakeBackend("piper", fail_load=True), FakeBackend("sapi")])
    assert syn.backend.name == "sapi" and fb.records[0].selected == "sapi"


def test_factory_refuses_cloud_when_not_allowed():
    cfg = load_config("cpu")
    cfg.tts.voices["edge"] = "en-US-AriaNeural"
    cfg.tts.fallback_backends = ["edge"]   # bypasses validation on purpose: the factory re-checks
    with pytest.raises(TtsError, match="not permitted"):
        create_synthesizer(cfg)


def test_piper_voice_path():
    assert piper_voice_path("en_US-joe-medium") == "en/en_US/joe/medium/en_US-joe-medium"
    with pytest.raises(TtsError):
        piper_voice_path("joe")


class _Eng:
    engine_name, model_id, load_info = "fake", "fake", {}

    def translate_batch(self, sources, prefixes=None, *, beam_size=4, max_tokens=256):
        lex = {"Habari za asubuhi.": "Good morning.", "Leo tunaenda sokoni.": "Today we go to the market."}
        return [lex.get(sources[0], "Hello of")]


def test_stage_speaks_finals_only_and_traces_events():
    from src.asr.streaming import AsrUpdate
    rec = EventRecorder()
    cfg = MtConfig(model="fake", translate_partials=True, partial_min_new_words=1)
    mt = MtStage(Translator(_Eng(), cfg), cfg, recorder=rec)
    b = FakeBackend()
    b.load()
    tts = TtsStage(Synthesizer([b], CFG), rec)
    out = []
    partial = AsrUpdate("partial", 0, "habari za", stable="habari za")
    for u in mt.on_asr_update(partial):
        out += tts.on_mt_update(u)
    assert out == [] and b.calls == []                        # provisional MT is never spoken
    final = AsrUpdate("final", 0, "habari za asubuhi. leo tunaenda sokoni", extra={"vad_reason": "silence"})
    for u in mt.on_asr_update(final):
        out += tts.on_mt_update(u)
    assert [u.chunk.text for u in out] == ["Good morning.", "Today we go to the market."]
    assert out[0].is_first and out[0].is_last
    types = [e.type for e in rec.events]
    assert types.count(EventType.TTS_FIRST_AUDIO) == 2 and types.count(EventType.TTS_COMPLETE) == 2
    assert all(e.audio is None for e in rec.events)            # history does not retain audio
    assert types.index(EventType.MT_FINAL) < types.index(EventType.TTS_CHUNK_REQUESTED)
