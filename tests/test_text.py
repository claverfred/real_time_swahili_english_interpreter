import pytest

from src.config.schema import TextConfig
from src.text import (
    Glossary,
    GlossaryEntry,
    SentenceAggregator,
    check_preservation,
    split_sentences,
    stabilize,
)
from src.text.entities import acronyms, canonical_number, mask, names, numbers, unmask
from src.text.glossary import GlossaryError


@pytest.mark.parametrize("raw,expected,change", [
    ("habari za asubuhi", "Habari za asubuhi.", "final_punct_added"),
    ("je unaenda sokoni", "Je unaenda sokoni?", "final_punct_added"),
    ("Habari , rafiki", "Habari, rafiki.", "punctuation_spacing"),
    ("ndiyo,sawa.", "Ndiyo, sawa.", "punctuation_spacing"),
    ("... habari yako", "Habari yako.", "edge_ellipsis_removed"),
    ("sawa sawa sawa sawa sawa", "Sawa sawa.", "repeat_collapsed"),
    ("Kweli!!", "Kweli!", "punctuation_spacing"),
])
def test_stabilize(raw, expected, change):
    st = stabilize(raw)
    assert st.text == expected and change in st.changes and st.original == raw


def test_stabilize_never_adds_words_and_respects_partial():
    st = stabilize("leo ninaenda", final=False)
    assert st.text == "Leo ninaenda"
    assert stabilize("bei ni 2.5", TextConfig(capitalize=False, add_final_punct=False)).text == "bei ni 2.5"
    assert stabilize("   ").text == ""


def test_split_sentences_swahili_abbreviations():
    text = "Dkt. Mwinyi alifika jana. Bw. Juma alisema ndiyo! Je, utakuja? J. Nyerere alikuwa rais."
    assert split_sentences(text) == [
        "Dkt. Mwinyi alifika jana.", "Bw. Juma alisema ndiyo!", "Je, utakuja?", "J. Nyerere alikuwa rais."]
    assert split_sentences("Bei ni 2.5 kwa kilo. Sawa") == ["Bei ni 2.5 kwa kilo.", "Sawa"]


def test_aggregator_holds_fragment_after_forced_split():
    agg = SentenceAggregator()
    units = agg.push("Habari za asubuhi. Leo tunaenda", segment_id=0, start_s=0, end_s=5, boundary=False)
    assert [u.text for u in units] == ["Habari za asubuhi."] and agg.pending == "Leo tunaenda"
    units = agg.push("sokoni kununua matunda.", segment_id=1, start_s=5, end_s=8, boundary=True)
    assert [u.text for u in units] == ["Leo tunaenda sokoni kununua matunda."]
    assert units[0].segment_ids == [0, 1] and units[0].start_s == 0 and units[0].complete
    assert agg.pending == ""


def test_aggregator_releases_incomplete_at_pause_and_flush():
    agg = SentenceAggregator()
    u = agg.push("ndiyo sawa", segment_id=0, start_s=0, end_s=1, boundary=True)
    assert u[0].text == "ndiyo sawa" and not u[0].complete and u[0].reason == "utterance_end"
    agg.push("na kisha", segment_id=1, start_s=1, end_s=2, boundary=False)
    f = agg.flush(3.0)
    assert f[0].text == "na kisha" and f[0].reason == "flush" and agg.flush() == []


def test_aggregator_caps_runon_at_clause():
    agg = SentenceAggregator(TextConfig(max_sentence_words=8))
    text = "tulienda sokoni asubuhi na mapema, tukanunua matunda mengi sana kwa ajili ya sherehe kubwa"
    units = agg.push(text, segment_id=0, start_s=0, end_s=6, boundary=False)
    assert units[0].text == "tulienda sokoni asubuhi na mapema," and units[0].reason == "max_words"
    assert all(len(u.text.split()) <= 8 for u in units) and len(agg.pending.split()) <= 8


def test_aggregator_uncertainty_propagates():
    agg = SentenceAggregator()
    agg.push("leo", segment_id=0, start_s=0, end_s=1, uncertain=True, boundary=False)
    u = agg.push("tunaenda.", segment_id=1, start_s=1, end_s=2, boundary=True)
    assert u[0].uncertain


def test_numbers_and_preservation():
    assert canonical_number("2,000") == canonical_number("2000") == "2000"
    assert canonical_number("2,5") == "2.5" and canonical_number("10%") == "10%"
    assert numbers("Watu 2,500 na asilimia 10% mwaka 2024.") == ["2500", "10%", "2024"]
    assert acronyms("CCM na TANESCO walikutana na WHO") == ["CCM", "TANESCO", "WHO"]
    assert names("Rais wa Tanzania alikwenda Dodoma.") == ["Tanzania", "Dodoma"]
    p = check_preservation("Watu 2500 kutoka CCM walifika Dodoma.", "Some 2,500 people arrived in Dodama.")
    assert p["numbers"]["missing"] == [] and p["acronyms"]["missing"] == ["CCM"]
    assert p["names"]["missing"] == ["Dodoma"]


def test_mask_unmask_roundtrip_and_failure():
    m = mask("Bei ni 2500 na 300", [(7, 11, "2500"), (15, 18, "300")])
    assert m.text == "Bei ni Z1Q na Z2Q" and m.mapping == {"Z1Q": "2500", "Z2Q": "300"}
    assert unmask("The price is Z1Q and Z2Q", m.mapping) == ("The price is 2500 and 300", [])
    assert unmask("The price is Z1Q", m.mapping)[1] == ["Z2Q"]
    big = mask("a b", [(0, 1, "A")] + [(2, 3, "B")])
    assert unmask("Z1Q Z2Q", big.mapping)[0] == "A B"


def test_glossary_load_find_and_check(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("source,target,kind\nBunge,Parliament,term\nWizara ya Afya,Ministry of Health,term\n"
                 "Mwananchi,,dnt\n", encoding="utf-8")
    g = Glossary.load(p)
    assert len(g) == 3
    hits = g.find("Wizara ya Afya ilizungumza na Bunge kuhusu Mwananchi")
    assert [e.target for _, _, e in hits] == ["Ministry of Health", "Parliament", "Mwananchi"]
    assert g.missing_targets("Bunge lilikutana", "The parliament met") == []
    assert g.missing_targets("Bunge lilikutana", "The assembly met") == ["Parliament"]
    bad = tmp_path / "b.csv"
    bad.write_text("source,target\nBunge,\n", encoding="utf-8")
    with pytest.raises(GlossaryError):
        Glossary.load(bad)
    assert Glossary([GlossaryEntry("UN", "UN", "dnt", case_sensitive=True)]).find("un UN") [0][0] == 3
