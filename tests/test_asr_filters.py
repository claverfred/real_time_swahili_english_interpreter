import pytest

from src.asr import AsrSegment, collapse_repeats, filter_segments, merge_overlap, normalize
from src.asr.filters import common_prefix, is_hallucination_phrase
from src.config.schema import AsrConfig

CFG = AsrConfig(model="x")


def seg(text, logp=-0.3, nsp=0.1, cr=1.5):
    return AsrSegment(0.0, 1.0, text, logp, nsp, cr)


def test_normalize_handles_swahili_punctuation_and_unicode():
    assert normalize("  Habari, Je? «Ndiyo»!  ") == "habari je ndiyo"
    assert normalize("Ng'ombe") == "ng ombe"
    assert normalize("café") == normalize("café")   # NFC


@pytest.mark.parametrize("text,expected", [
    ("na na na na na", "na na"),
    ("asante sana asante sana asante sana asante sana", "asante sana asante sana"),
    ("ni ni", "ni ni"),
    ("Habari yako rafiki", "Habari yako rafiki"),
    ("leo leo leo ninaenda", "leo leo ninaenda"),
    ("kwa hiyo sawa sawa sawa sawa kabisa", "kwa hiyo sawa sawa kabisa"),   # unaligned loop
])
def test_collapse_repeats(text, expected):
    assert collapse_repeats(text) == expected


def test_filter_rules_and_reasons():
    segs = [
        seg(""),
        seg("Habari", logp=-1.5, nsp=0.9),
        seg("ndiyo ndiyo", cr=3.0),
        seg("Asante kwa kutazama!"),
        seg("Ninaenda sokoni."),
        seg("ninaenda sokoni"),                 # duplicate of previous
        seg("maneno yasiyo wazi", logp=-1.8),  # very low confidence
        seg("labda hivi", logp=-1.2),          # uncertain but kept
    ]
    filter_segments(segs, CFG)
    assert [s.rejected for s in segs] == [
        "empty", "no_speech", "repetition_loop", "hallucination_phrase", None,
        "duplicate_segment", "very_low_confidence", None,
    ]
    assert segs[7].uncertain and not segs[4].uncertain


def test_duplicate_of_previous_utterance_is_dropped():
    s = [seg("Habari za asubuhi.")]
    filter_segments(s, CFG, previous_text="habari za asubuhi")
    assert s[0].rejected == "duplicate_segment"


def test_hallucination_phrase_requires_match():
    assert is_hallucination_phrase("Thanks for watching!")
    assert not is_hallucination_phrase("Asante kwa chakula")


def test_merge_overlap():
    assert merge_overlap("leo tunaenda sokoni", "sokoni kununua matunda") == "kununua matunda"
    assert merge_overlap("habari za asubuhi", "Za asubuhi, rafiki") == "rafiki"
    assert merge_overlap("habari", "nzuri sana") == "nzuri sana"
    assert merge_overlap("", "nzuri") == "nzuri"


def test_common_prefix_normalized():
    assert common_prefix("Habari, za asubuhi".split(), "habari za jioni".split()) == 2
