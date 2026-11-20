from src.asr.streaming import AsrUpdate
from src.asr.types import AsrResult
from src.config.schema import MtConfig, TextConfig
from src.mt import MtStage, Translator
from src.mt.langs import family_of, lang_code
from src.pipeline.events import EventRecorder, EventType
from src.text import Glossary, GlossaryEntry

LEX = {"habari": "hello", "za": "of", "asubuhi": "morning", "leo": "today", "tunaenda": "we-go",
       "sokoni": "market", "bei": "price", "ni": "is", "shilingi": "shillings", "bunge": "assembly"}


class FakeEngine:
    """Word-by-word 'translation'; tokens it does not know pass through (like names, numbers)."""
    engine_name, model_id = "fake", "fake-mt"

    def __init__(self, drop_placeholders=False, runaway=False):
        self.calls = []
        self.drop_placeholders, self.runaway = drop_placeholders, runaway
        self.load_info = {"engine": "fake"}

    def translate_batch(self, sources, prefixes=None, *, beam_size=4, max_tokens=256):
        self.calls.append((list(sources), list(prefixes) if prefixes else None, beam_size))
        out = []
        for i, s in enumerate(sources):
            words = [LEX.get(w.strip(".,?!").lower(), w.strip(".,?!")) for w in s.split()]
            if self.drop_placeholders:
                words = [w for w in words if not (w.startswith("Z") and w.endswith("Q"))]
            if self.runaway:
                words += ["and", "then"] * 20
            text = " ".join(words)
            if prefixes and prefixes[i]:  # context: model returns the continuation only
                n_ctx = len(prefixes[i].split())
                text = " ".join(text.split()[n_ctx:])
            out.append(text)
        return out

    def warmup(self):
        return 1.0


def make(cfg=None, **kw):
    eng = FakeEngine(**kw)
    return Translator(eng, cfg or MtConfig(model="fake-mt")), eng


def test_family_and_codes():
    assert family_of("facebook/nllb-200-distilled-600M") == "nllb"
    assert family_of("Helsinki-NLP/opus-mt-swc-en") == "marian"
    assert family_of("facebook/m2m100_418M") == "m2m"
    assert lang_code("m2m", "swh_Latn") == "sw" and lang_code("marian", "swh_Latn") is None


def test_translate_basic_cache_and_flags():
    tr, eng = make()
    r = tr.translate("Habari za asubuhi.")
    assert r.text == "hello of morning" and not r.cached and r.flags == []
    r2 = tr.translate("Habari  za asubuhi.")
    assert r2.cached and len(eng.calls) == 1 and tr.cache_hits == 1
    assert "incomplete_source" in tr.translate("leo tunaenda").flags
    assert tr.translate("").flags == ["empty_source"]


def test_preservation_flags():
    tr, _ = make()
    r = tr.translate("Bei ni 2500 Dodoma.")
    assert r.preservation["numbers"]["missing"] == [] and "number_missing" not in r.flags


def test_runaway_output_is_collapsed_and_flagged():
    tr, _ = make(runaway=True)
    r = tr.translate("Habari.")
    assert "repeat_collapsed" in r.flags and r.text.count("and then") <= 2


def test_mask_strategy_and_failure_retry():
    cfg = MtConfig(model="fake-mt", entity_strategy="mask")
    tr, eng = make(cfg)
    r = tr.translate("Bei ni 2,500 shilingi.")
    assert eng.calls[0][0] == ["Bei ni Z1Q shilingi."] and "2,500" in r.text
    tr2, eng2 = make(cfg, drop_placeholders=True)
    r2 = tr2.translate("Bei ni 2500.")
    assert "mask_failed" in r2.flags and eng2.calls[-1][0] == ["Bei ni 2500."]


def test_glossary_mask_inserts_target():
    cfg = MtConfig(model="fake-mt", entity_strategy="mask")
    g = Glossary([GlossaryEntry("Bunge", "Parliament")])
    tr = Translator(FakeEngine(), cfg, g)
    r = tr.translate("Bunge leo.")
    assert r.text.startswith("Parliament") and "glossary_missing" not in r.flags
    tr_verify = Translator(FakeEngine(), MtConfig(model="fake-mt"), g)
    assert "glossary_missing" in tr_verify.translate("Bunge leo.").flags


def test_context_prefix_mode():
    cfg = MtConfig(model="fake-mt", context_mode="prefix", context_sentences=1)
    tr, eng = make(cfg)
    r = tr.translate("Leo tunaenda sokoni.", context=[("Habari za asubuhi.", "hello of morning")])
    assert eng.calls[0][0] == ["Habari za asubuhi. Leo tunaenda sokoni."]
    assert eng.calls[0][1] == ["hello of morning"]
    assert r.text == "today we-go market" and r.context_used == 1
    assert make()[0].translate("Leo.", context=[("a", "b")]).context_used == 0   # mode none


def test_provisional_uses_greedy_and_no_context():
    cfg = MtConfig(model="fake-mt", context_mode="prefix", beam_size=5)
    tr, eng = make(cfg)
    r = tr.translate("Leo tunaenda", context=[("a", "b")], provisional=True)
    assert r.provisional and r.context_used == 0 and eng.calls[0][2] == 1


def _final(seg, text, reason="silence", start=0.0, end=1.0, uncertain=False):
    res = AsrResult(text, "sw", True, end - start, 10.0, "fake")
    up = AsrUpdate("final", seg, text, result=res, start_s=start, end_s=end, extra={"vad_reason": reason})
    if uncertain:
        up.extra["uncertain"] = True
        up.result = None
    return up


def test_stage_finals_split_context_and_events():
    rec = EventRecorder()
    tr, eng = make(MtConfig(model="fake-mt", context_mode="prefix", context_sentences=2))
    stage = MtStage(tr, tr.cfg, TextConfig(), rec)
    ups = stage.on_asr_update(_final(0, "habari za asubuhi leo tunaenda", reason="max_duration", end=20))
    assert ups == []                               # fragment held across the forced split
    ups = stage.on_asr_update(_final(1, "sokoni. bei ni 2500", start=20, end=24))
    assert [u.source for u in ups] == ["Habari za asubuhi leo tunaenda sokoni.", "Bei ni 2500."]
    assert ups[1].result.context_used == 1
    types = [e.type for e in rec.events]
    assert types.count(EventType.TRANSCRIPT_STABILIZED) == 2
    assert types.count(EventType.MT_REQUESTED) == 2 and types.count(EventType.MT_FINAL) == 2
    assert rec.of_type(EventType.MT_FINAL)[0].data["source"] == ups[0].source


def test_stage_partials_only_from_stable_prefix():
    cfg = MtConfig(model="fake-mt", translate_partials=True, partial_min_new_words=2)
    tr, eng = make(cfg)
    stage = MtStage(tr, cfg, TextConfig(), EventRecorder())
    p = lambda stable, unstable="": AsrUpdate("partial", 0, f"{stable} {unstable}".strip(), stable=stable,  # noqa: E731
                                             unstable=unstable)
    assert stage.on_asr_update(p("", "habari za")) == []
    assert stage.on_asr_update(p("habari", "za asubuhi")) == []          # 1 new word < 2
    out = stage.on_asr_update(p("habari za", "asubuhi leo"))
    assert out[0].kind == "partial" and out[0].source == "habari za" and out[0].result.provisional
    assert stage.on_asr_update(p("habari za asubuhi")) == []
    assert all("leo" not in c[0][0] for c in eng.calls)                    # unstable tail never sent
    assert stage.history == []


def test_stage_uncertain_skip_policy():
    cfg = MtConfig(model="fake-mt", uncertain_policy="skip")
    tr, eng = make(cfg)
    stage = MtStage(tr, cfg)
    ups = stage.on_asr_update(_final(0, "labda hivi", uncertain=True))
    assert ups[0].kind == "skipped" and eng.calls == []
    flagged = MtStage(*make()[:1], MtConfig(model="fake-mt"))
    assert flagged.on_asr_update(_final(0, "labda hivi", uncertain=True))[0].result.uncertain


def test_stage_flush_releases_pending():
    tr, _ = make()
    stage = MtStage(tr, tr.cfg)
    stage.on_asr_update(_final(0, "leo tunaenda", reason="max_duration"))
    ups = stage.flush(5.0)
    assert ups[0].source == "Leo tunaenda" and "incomplete_source" in ups[0].result.flags
