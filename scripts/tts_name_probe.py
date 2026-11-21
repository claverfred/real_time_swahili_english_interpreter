"""Pick respellings for the pronunciation dictionary: synthesize each candidate spelling in a
carrier sentence, transcribe it back with Whisper (English), and count exact name recoveries.

    python scripts/tts_name_probe.py --candidates piper:en_US-lessac-medium sapi:default

A spelling "wins" for a voice when Whisper writes the *original* name back. This is a machine
proxy (Whisper may not know rare names), so it is used to reject bad respellings, not to prove
good pronunciation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr.faster_whisper_backend import FasterWhisperAsr  # noqa: E402
from src.audio.io import resample  # noqa: E402
from src.config import load_config  # noqa: E402
from src.config.schema import AsrConfig  # noqa: E402
from src.platform.resolve import resolve_runtime  # noqa: E402
from src.tts import create_synthesizer  # noqa: E402

VARIANTS = {
    "Dodoma": ["Dodoma", "doh-doh-mah", "Doedoma"],
    "Suluhu": ["Suluhu", "soo-loo-hoo", "Sooloohoo"],
    "Samia": ["Samia", "sah-mee-ah", "Sameeah"],
    "Nyerere": ["Nyerere", "nyeh-reh-reh", "Nyehrehreh"],
    "Mbeya": ["Mbeya", "mbay-ah", "Mbayah"],
    "Iringa": ["Iringa", "ee-reen-gah", "Eereenga"],
    "Morogoro": ["Morogoro", "moh-roh-goh-roh", "Morohgoroh"],
    "Mwinyi": ["Mwinyi", "mween-yee", "Mweenyee"],
    "Magufuli": ["Magufuli", "mah-goo-foo-lee", "Magoofoolee"],
    "TANESCO": ["TANESCO", "tah-ness-koh", "Tanesko"],
}
CARRIER = "Yesterday {} was in the news again."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cpu")
    ap.add_argument("--candidates", nargs="+", required=True)
    args = ap.parse_args()
    cfg0, _, fb = resolve_runtime(load_config(args.profile))
    asr = FasterWhisperAsr(AsrConfig(model="small", language="en", beam_size=5), model="small",
                           device=cfg0.environment.device, compute_type=cfg0.asr.compute_type,
                           cpu_threads=cfg0.environment.cpu_threads, cache_dir=cfg0.storage.cache_dir)
    asr.load()
    report: dict[str, dict[str, list[str]]] = {}
    for cand in args.candidates:
        backend, _, voice = cand.partition(":")
        cfg = cfg0.model_copy(deep=True)
        cfg.tts.backend, cfg.tts.voice, cfg.tts.fallback_backends = backend, voice, []
        cfg.tts.spell_acronyms = False   # test spellings exactly as written
        syn = create_synthesizer(cfg, fb)
        report[cand] = {}
        for name, spellings in VARIANTS.items():
            wins = []
            for sp in spellings:
                res = syn.synthesize(CARRIER.format(sp))
                hyp = asr.transcribe(resample(res.audio, res.sample_rate, 16000)).text
                ok = re.search(rf"\b{re.escape(name)}\b", hyp, re.IGNORECASE) is not None
                print(f"{cand:<28} {name:<10} {sp:<18} {'OK ' if ok else '-- '} {hyp}", file=sys.stderr)
                if ok:
                    wins.append(sp)
            report[cand][name] = wins
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
