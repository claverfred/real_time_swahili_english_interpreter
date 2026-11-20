"""Real-model MT checks. Slow; run with ``pytest -m slow``. Skipped unless the converted
CTranslate2 model is already in cache/ (no downloads during tests)."""

from pathlib import Path

import pytest

from src.config.schema import MtConfig
from src.mt import Translator, make_engine
from src.mt.convert import find_converted
from src.platform.cpu import auto_threads

pytestmark = pytest.mark.slow
CACHE = Path("cache")
NLLB = "facebook/nllb-200-distilled-600M"


@pytest.fixture(scope="module")
def nllb():
    if find_converted(CACHE, NLLB, "int8") is None:
        pytest.skip("converted NLLB not cached; run scripts/mt_eval.py once")
    eng = make_engine("ctranslate2", model_id=NLLB, family="nllb", src_lang="swh_Latn", tgt_lang="eng_Latn",
                      compute_type="int8", threads=auto_threads(0), cache_root=CACHE, local_files_only=True)
    eng.load()
    return eng


def test_nllb_translates_and_preserves_numbers(nllb):
    tr = Translator(nllb, MtConfig(model=NLLB, beam_size=2))
    r = tr.translate("Bei ya nyanya ni shilingi 2500 kwa kilo moja.")
    assert "tomato" in r.text.lower() and ("2,500" in r.text or "2500" in r.text)
    assert "number_missing" not in r.flags and r.latency_ms > 0


def test_nllb_context_prefix_returns_only_new_sentence(nllb):
    tr = Translator(nllb, MtConfig(model=NLLB, beam_size=2, context_mode="prefix", context_sentences=1))
    first = tr.translate("Habari za asubuhi.")
    second = tr.translate("Leo ninaenda sokoni.", context=[("Habari za asubuhi.", first.text)])
    assert second.context_used == 1
    assert "market" in second.text.lower() and "morning" not in second.text.lower()


def test_nllb_mask_roundtrip(nllb):
    tr = Translator(nllb, MtConfig(model=NLLB, beam_size=2, entity_strategy="mask"))
    r = tr.translate("Watu 2,345 walihudhuria mkutano.")
    assert "2,345" in r.text and "mask_failed" not in r.flags
