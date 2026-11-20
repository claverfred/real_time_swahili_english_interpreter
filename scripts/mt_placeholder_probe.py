"""Which placeholder forms survive translation unchanged? Decides src/text/entities.placeholder.

    python scripts/mt_placeholder_probe.py --models facebook/nllb-200-distilled-600M Helsinki-NLP/opus-mt-swc-en

Each template is substituted for the number in Swahili sentences and counted as surviving
when it appears exactly once in the English output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mt.engines import make_engine  # noqa: E402
from src.mt.langs import family_of  # noqa: E402
from src.platform.cpu import auto_threads  # noqa: E402

SENTENCES = [
    "Bei ya nyanya ni shilingi {} kwa kilo moja.",
    "Watu {} walihudhuria mkutano jana jioni.",
    "Treni inaondoka saa {} asubuhi kutoka Dodoma.",
    "Shule hiyo ilijengwa mwaka {} na serikali.",
    "Mvua ilinyesha kwa siku {} mfululizo katika mkoa wa Mbeya.",
    "Alipata alama {} kati ya mia moja kwenye mtihani.",
]
TEMPLATES = ["X{}", "Z{}Q", "[{}]", "<{}>", "__{}__", "#{}", "NUM{}", "{}{}{}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["facebook/nllb-200-distilled-600M"])
    ap.add_argument("--engine", default="ctranslate2")
    args = ap.parse_args()
    report = {}
    for model in args.models:
        eng = make_engine(args.engine, model_id=model, family=family_of(model), src_lang="swh_Latn",
                          tgt_lang="eng_Latn", compute_type="int8", threads=auto_threads(0))
        print(f"== {model}: {json.dumps(eng.load())}", file=sys.stderr)
        plain = eng.translate_batch([s.format("2500") for s in SENTENCES], beam_size=4)
        for s, t in zip(SENTENCES, plain, strict=True):
            print(f"   {s.format('2500'):<60} -> {t}", file=sys.stderr)
        scores = {}
        for tpl in TEMPLATES:
            ph = tpl.format(*[1] * tpl.count("{}"))
            outs = eng.translate_batch([s.format(ph) for s in SENTENCES], beam_size=4)
            ok = sum(o.count(ph) == 1 for o in outs)
            scores[tpl] = f"{ok}/{len(SENTENCES)}"
            print(f"   {tpl:<8} {ok}/{len(SENTENCES)}  e.g. {outs[0]}", file=sys.stderr)
        report[model] = scores
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
