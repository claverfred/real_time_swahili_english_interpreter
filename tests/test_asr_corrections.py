from pathlib import Path

import pytest

from src.asr.corrections import AsrCorrections, Correction, CorrectionsError
from src.config import load_config

REPO = Path(__file__).resolve().parents[1]


def test_apply_whole_word_longest_first_keeps_capital():
    c = AsrCorrections([Correction("barimbali", "mbalimbali"), Correction("madam barimbali", "mambo mbalimbali"),
                        Correction("mwenye kiti", "mwenyekiti")])
    text, n = c.apply("Madam barimbali na barimbali, Mwenye kiti.")
    assert text == "Mambo mbalimbali na mbalimbali, Mwenyekiti."
    assert n == 3
    # whole words only: no change inside a longer word
    assert c.apply("wabarimbalini") == ("wabarimbalini", 0)
    assert AsrCorrections().apply("habari") == ("habari", 0)


def test_load_skips_comments_and_validates(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("# comment\nheard,correct,note\nsabari ya,habari ya,greeting\n\n", encoding="utf-8")
    c = AsrCorrections.load(f)
    assert len(c) == 1 and c.apply("Sabari ya jioni.") == ("Habari ya jioni.", 1)
    f.write_text("heard,correct\nx,\n", encoding="utf-8")
    with pytest.raises(CorrectionsError):
        AsrCorrections.load(f)
    f.write_text("heard,correct\nx,y\nX,z\n", encoding="utf-8")
    with pytest.raises(CorrectionsError, match="repeats"):
        AsrCorrections.load(f)
    with pytest.raises(CorrectionsError, match="not found"):
        AsrCorrections.load(tmp_path / "missing.csv")


def test_shipped_list_loads_and_fixes_session_errors():
    cfg = load_config("cpu")
    c = AsrCorrections.load(REPO / cfg.asr.corrections_path)
    assert len(c) >= 10
    assert c.apply("Mvishimiwa mwenye kiti.")[0] == "Mheshimiwa mwenyekiti."


class _FixedAsr:
    model_name = "fixed"

    def transcribe(self, audio, *, fast=False, initial_prompt=None, previous_text="", word_timestamps=None):
        from src.asr import AsrResult
        return AsrResult("mwenye kiti wa kikao", "sw", True, len(audio) / 16000, 5.0, "fixed")


def test_streaming_final_applies_corrections():
    import numpy as np

    from src.asr import StreamingTranscriber
    from src.config.schema import AsrConfig, VadConfig
    from src.pipeline.events import EventRecorder, EventType
    from src.vad import VadSegmenter
    from tests.test_asr_streaming import ScriptedVad

    rec = EventRecorder()
    probs = [0] * 20 + [1] * 100 + [0] * 60
    seg = VadSegmenter(ScriptedVad(probs), VadConfig(min_speech_ms=30, min_silence_ms=300, speech_pad_ms=0,
                                                     hangover_ms=0))
    st = StreamingTranscriber(_FixedAsr(), AsrConfig(model="fixed"), 16000, rec, partials=False,
                              corrections=AsrCorrections([Correction("mwenye kiti", "mwenyekiti")]))
    evs = list(seg.process(np.zeros(len(probs) * 160, np.float32))) + list(seg.flush())
    finals = [u for ev in evs for u in st.on_vad_event(ev) if u.kind == "final"]
    assert finals[0].text == "mwenyekiti wa kikao"
    assert finals[0].extra["original"] == "mwenye kiti wa kikao"
    ev = next(e for e in rec.events if e.type == EventType.ASR_FINAL)
    assert ev.data["corrections"] == 1 and ev.data["original"] == "mwenye kiti wa kikao"
    assert st.corrections_applied == 1
