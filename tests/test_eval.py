"""Phase 8 evaluation: metrics, Swahili number normalization, perturbations, datasets, runners."""

import asyncio
import csv
import json

import numpy as np
import pytest

from src.audio import save_wav
from src.config import load_config
from src.eval import metrics as M
from src.eval.datasets import SpeechItem, load_fleurs, load_flores
from src.eval.human import make_sheet, score_sheets
from src.eval.perturb import CONDITIONS, apply
from src.eval.speech import build_stream, run_mt, run_speech, score_speech
from src.mt import Translator
from src.pipeline import Components
from tests.test_asr_streaming import FakeAsr
from tests.test_interpreter import LevelVad, speech
from tests.test_mt import FakeEngine

pytest.importorskip("jiwer")
pytest.importorskip("sacrebleu")
SR = 16000


# -- metrics -------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("katika miaka yote ya elfu moja mia tisa sitini", "katika miaka yote ya 1960"),
    ("shilingi elfu mbili kwa kilo moja", "shilingi 2000 kwa kilo moja"),
    ("karne ya tatu", "karne ya tatu"),                       # lone units are left alone
    ("watu mia tatu na hamsini", "watu 350"),
    ("kumi na tano", "15"),
    ("laki tatu", "300000"),
    ("mia na", "100 na"),                                     # trailing conjunction kept
    ("elfu mia tatu thelathini", "330000"),                   # hundreds multiplier after elfu
    ("gigahezi mbili nukta nne", "gigahezi 2.4"),             # spoken decimal
    ("mia nane na mbili nukta moja moja", "802.11"),
    ("tatu nukta", "tatu nukta"),                             # no fraction: not a decimal
    ("elfu kumi nukta", "10000 nukta"),
])
def test_swahili_number_words_to_digits(text, expected):
    assert M.sw_numbers_to_digits(text) == expected


def test_number_words_to_digits_keeps_punctuation_and_stops_at_it():
    from src.text.sw_numbers import number_words_to_digits

    assert number_words_to_digits("Katika miaka ya elfu moja mia tisa sitini, Brzezinski.") == \
        ("Katika miaka ya 1960, Brzezinski.", 1)
    assert number_words_to_digits("Sitini.") == ("60.", 1)
    assert number_words_to_digits("kumi, na tano") == ("10, na tano", 1)    # comma ends the phrase
    assert number_words_to_digits("karne ya tatu") == ("karne ya tatu", 0)


def test_stabilize_number_option_is_off_by_default_and_listed_when_on():
    from src.config.schema import TextConfig
    from src.text import stabilize

    off = stabilize("bei ni shilingi elfu mbili", TextConfig())
    on = stabilize("bei ni shilingi elfu mbili", TextConfig(number_words_to_digits=True))
    assert off.text == "Bei ni shilingi elfu mbili."
    assert on.text == "Bei ni shilingi 2000." and "number_words_to_digits" in on.changes


def test_final_decode_temperatures_come_from_config():
    from src.asr.faster_whisper_backend import FasterWhisperAsr
    from src.config.schema import AsrConfig

    seen = {}

    class Model:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return iter([]), type("Info", (), {"language": "sw"})()

    asr = FasterWhisperAsr(AsrConfig(model="x", temperatures=[0.0]))
    asr._model = Model()
    asr.transcribe(np.zeros(SR, np.float32))
    assert seen["temperature"] == (0.0,)
    asr.transcribe(np.zeros(SR, np.float32), fast=True)
    assert seen["temperature"] == 0.0


def test_number_normalized_wer_removes_format_mismatch():
    ref, hyp = "Miaka ya 1960.", "miaka ya elfu moja mia tisa sitini"
    assert M.wer_cer([ref], [hyp])["wer"] > 0.5
    norm = [M.sw_numbers_to_digits(M.normalize(x)) for x in (ref, hyp)]
    assert M.wer_cer([norm[0]], [norm[1]])["wer"] == 0.0


def test_recall_numbers_and_names():
    refs = ["Bei ni 2,000 na Juma alikuja.", "Hakuna nambari hapa."]
    hyps = ["bei ni 2000 na juma alikuja", "hakuna"]
    num = M.recall(refs, hyps, "numbers")
    assert (num["recall"], num["items"]) == (1.0, 1)
    names = M.recall(refs, hyps, "names")
    assert names["recall"] == 1.0 and names["items"] == 1     # 'Bei' starts the sentence: not a name
    assert M.recall(["Hakuna."], ["x"], "numbers")["recall"] is None


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    errors, totals = [0, 1, 2, 5, 0, 3, 1, 0], [10] * 8
    a = M.bootstrap_ratio(errors, totals, n_boot=300, seed=1)
    b = M.bootstrap_ratio(errors, totals, n_boot=300, seed=1)
    assert a == b
    assert a["low"] <= sum(errors) / sum(totals) <= a["high"]


def test_length_ratio_and_mt_scores():
    rates = M.length_ratio_rates(["one two three four five"] * 2, ["one", "one two three four five"])
    assert rates["omission_rate"] == 0.5 and rates["addition_rate"] == 0.0
    sents = ["the cat sat on the mat today", "a big dog ran across the road"]
    s = M.mt_scores(sents, sents, bootstrap=20)
    assert s["bleu"] == pytest.approx(100.0) and s["chrf"] == pytest.approx(100.0)
    assert "nrefs:1" in s["signatures"]["bleu"] and len(s["ci95"]["chrf"]) == 2


# -- perturbations -------------------------------------------------------------------------------
def test_perturbations_are_deterministic_bounded_and_do_what_they_say():
    x = speech((0.3, False), (1.5, True), (0.3, False)) * 0.5
    for name in CONDITIONS:
        y = apply(name, x, "item1")
        assert y.dtype == np.float32 and np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0, name
        assert np.array_equal(y, apply(name, x, "item1")), name
    assert len(apply("fast_1.15x", x, "i")) < len(x) < len(apply("slow_0.85x", x, "i"))
    noisy = apply("noise_10db", x, "i")
    assert not np.array_equal(noisy, apply("noise_10db", x, "j"))     # seeded per item
    assert np.sqrt(np.mean(apply("quiet_-20db", x, "i") ** 2)) == pytest.approx(np.sqrt(np.mean(x ** 2)) / 10, rel=1e-3)
    with pytest.raises(ValueError):
        apply("underwater", x, "i")


# -- datasets ------------------------------------------------------------------------------------
def _fixture_data(tmp_path, clips):
    fl = tmp_path / "flores"
    fl.mkdir()
    (fl / "dev.swh_Latn.txt").write_text("Sentensi ya dev.\n", encoding="utf-8")
    (fl / "dev.eng_Latn.txt").write_text("A dev sentence.\n", encoding="utf-8")
    (fl / "devtest.swh_Latn.txt").write_text("Leo ninaenda sokoni.\nLeo ninaenda sokoni kununua matunda.\n",
                                             encoding="utf-8")
    (fl / "devtest.eng_Latn.txt").write_text("Today I go to market.\nToday I am going to the market to buy fruit.\n",
                                             encoding="utf-8")
    fs = tmp_path / "fleurs"
    (fs / "test").mkdir(parents=True)
    rows = []
    for i, (text, audio) in enumerate(clips):
        name = f"{i:03d}.wav"
        save_wav(fs / "test" / name, audio, SR)
        rows.append({"id": str(i), "file": name, "raw": text, "gender": "FEMALE" if i % 2 else "MALE",
                     "path": str(fs / "test" / name)})
    (fs / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (fs / "manifest.json").write_text(json.dumps({"name": "fleurs_fixture", "license": "CC-BY-4.0"}), encoding="utf-8")
    return fs, fl


def test_load_fleurs_links_english_by_text_and_reports_unmatched(tmp_path):
    tone = speech((0.2, False), (1.0, True), (0.2, False))
    fs, fl = _fixture_data(tmp_path, [("Leo, ninaenda sokoni!", tone), ("Haipo kwenye FLORES.", tone)])
    items, prov = load_fleurs(fs, n=0, flores_root=fl)
    assert items[0].en_ref == "Today I go to market." and items[0].meta["flores"] == "devtest:1"
    assert items[1].en_ref is None and prov["unmatched_english"] == 1
    src, ref, p = load_flores(fl, "devtest", n=1)
    assert (src, ref, p["items"]) == (["Leo ninaenda sokoni."], ["Today I go to market."], 1)


# -- runners with fake models --------------------------------------------------------------------
def _components(cfg):
    return Components(vad=LevelVad(), asr=FakeAsr(), translator=Translator(FakeEngine(), cfg.mt), synthesizer=None)


def _cfg():
    return load_config("cpu", ["vad.min_silence_ms=300", "vad.speech_pad_ms=0", "vad.hangover_ms=0",
                               "vad.min_speech_ms=60", "asr.corrections_path=null"], environ={})


def test_run_speech_scores_asr_and_speech_to_english(tmp_path):
    # FakeAsr emits one word per 0.5 s: 2.0 s of tone -> "leo ninaenda sokoni kununua"
    clip = speech((0.3, False), (2.0, True), (0.5, False))
    fs, fl = _fixture_data(tmp_path, [("Leo ninaenda sokoni kununua matunda.", clip),
                                      ("Leo ninaenda sokoni.", speech((0.3, False), (1.5, True), (0.5, False)))])
    items, _ = load_fleurs(fs, flores_root=fl)
    cfg = _cfg()
    seen = []
    rows = asyncio.run(run_speech(cfg, items, _components(cfg), progress=lambda k, n, r: seen.append(k)))
    assert seen == [1, 2] and len(rows) == 2
    assert rows[0]["sw_hyp"].lower().rstrip(".") == "leo ninaenda sokoni kununua"
    assert rows[1]["sw_hyp"].lower().rstrip(".") == "leo ninaenda sokoni"
    assert all(r["en_hyp"] and r["en_oracle"] for r in rows)
    out = score_speech(rows, bootstrap=50)
    assert out["asr"]["wer"] == pytest.approx(1 / 8)                   # one deleted word of 8
    assert out["asr"]["sentences_exact"] == 0.5
    assert set(out["asr"]["by_gender"]) == {"FEMALE", "MALE"}
    s2t = out["speech_to_english"]
    assert 0 <= s2t["chrf"] <= 100 and "signatures" in s2t and "chrf_by_asr_wer" in s2t
    assert "oracle_mt" in out and "asr_induced_chrf_loss" in out
    assert out["failure_examples"][0]["wer"] >= out["failure_examples"][-1]["wer"]
    assert out["errors"] == 0


def test_run_speech_under_noise_condition_runs(tmp_path):
    clip = speech((0.3, False), (2.0, True), (0.5, False))
    fs, fl = _fixture_data(tmp_path, [("Leo ninaenda sokoni kununua matunda.", clip)])
    items, _ = load_fleurs(fs, flores_root=fl)
    cfg = _cfg()
    rows = asyncio.run(run_speech(cfg, items, _components(cfg), condition="quiet_-20db", oracle=False))
    assert rows[0]["condition"] == "quiet_-20db" and "en_oracle" not in rows[0]


def test_build_stream_places_recordings_with_gaps(tmp_path):
    clip = speech((1.0, True))
    p = tmp_path / "a.wav"
    save_wav(p, clip, SR)
    items = [SpeechItem("a", str(p), "x"), SpeechItem("b", str(p), "y")]
    audio, spans = build_stream(items, gap_s=2.0, lead_s=1.0)
    assert len(audio) == SR * (1 + 1 + 2 + 1 + 2)
    assert spans[0] == {"item_id": "a", "start_s": 1.0, "end_s": 2.0}
    assert spans[1]["start_s"] == pytest.approx(4.0)               # lead 1 + clip 1 + gap 2


def test_run_mt_reports_quality_latency_and_preservation():
    cfg = _cfg()
    tr = Translator(FakeEngine(), cfg.mt)
    r = run_mt(tr, ["Leo ninaenda sokoni.", "Bei ni 2000."], ["Today I go to market.", "The price is 2000."],
               bootstrap=10)
    assert r["n"] == 2 and r["latency_ms"]["n"] == 2 and len(r["hypotheses"]) == 2
    assert r["numbers_kept_from_source"]["recall"] == 1.0


# -- human evaluation kit ------------------------------------------------------------------------
def test_human_sheet_hides_references_and_scorer_aggregates(tmp_path):
    rows = [{"item_id": f"u{i}", "sw_ref": f"Swahili {i}", "en_hyp": f"English {i}", "en_ref": f"Ref {i}",
             "sw_hyp": f"asr {i}", "wer": 0.0 if i % 2 else 0.4, "condition": "clean"} for i in range(6)]
    info = make_sheet(rows, tmp_path / "h", n=4, seed=1)
    assert info["items"] == 4
    with open(info["sheet"], encoding="utf-8-sig", newline="") as fh:
        sheet = list(csv.DictReader(fh))
    assert "english_reference" not in sheet[0] and "asr_text" not in sheet[0]
    for rater, (ade, flu) in {"r1": (5, 4), "r2": (4, 4)}.items():
        path = tmp_path / f"{rater}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sheet[0]))
            w.writeheader()
            for r in sheet:
                w.writerow({**r, "adequacy_1_5": ade, "fluency_1_5": flu, "names_ok_y_n_na": "y",
                            "error_severity_none_minor_major_critical": "minor"})
    s = score_sheets([tmp_path / "r1.csv", tmp_path / "r2.csv"], info["key"])
    assert s["scales"]["adequacy_1_5"]["mean"] == 4.5 and s["scales"]["adequacy_1_5"]["n"] == 8
    assert s["adequacy_agreement"][0]["within_1"] == 1.0 and s["adequacy_agreement"][0]["exact_agreement"] == 0.0
    assert s["scales"]["error_severity"]["minor"] == 1.0
    assert sum(v["n"] for v in s["adequacy_by_asr_quality"].values()) == 8
