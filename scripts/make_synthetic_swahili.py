"""Generate a small *synthetic* Swahili smoke-test set with Edge-TTS.

    python scripts/make_synthetic_swahili.py --out data/synthetic/sw_tts_v1

This is for integration testing only: TTS speech is far cleaner than real speech, so WER on it
is NOT a benchmark and must never be reported as Swahili ASR accuracy. Real evaluation uses
FLEURS / Common Voice in Phase 8.

The sentence texts below are the references. Only this public text is sent to the Edge-TTS
service; no user audio leaves the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import load_audio, save_wav  # noqa: E402

SENTENCES = [
    ("s01", "Habari za asubuhi. Leo ninaenda sokoni kununua matunda."),
    ("s02", "Bei ya nyanya ni shilingi elfu mbili kwa kilo moja."),
    ("s03", "Rais wa Tanzania alizungumza na wananchi mjini Dodoma jana jioni."),
    ("s04", "Wanafunzi wanapaswa kusoma kwa bidii ili wafaulu mitihani yao."),
    ("s05", "Mvua kubwa inatarajiwa kunyesha katika maeneo ya pwani wiki hii."),
]
VOICES = ["sw-KE-ZuriNeural", "sw-TZ-DaudiNeural"]
SR = 16000


async def synth(text: str, voice: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(path))


async def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "_mp3"
    raw.mkdir(exist_ok=True)
    items = []
    for voice in VOICES:
        for sid, text in SENTENCES:
            mp3 = raw / f"{sid}_{voice}.mp3"
            if not mp3.exists():
                await synth(text, voice, mp3)
            audio, _ = load_audio(mp3, SR)
            wav = save_wav(out / f"{sid}_{voice}.wav", audio, SR)
            items.append({"id": f"{sid}_{voice}", "audio": wav.name, "text": text, "voice": voice,
                          "duration_s": round(len(audio) / SR, 3),
                          "sha256": hashlib.sha256(wav.read_bytes()).hexdigest()})

    # One long clip: all sentences of the first voice separated by 1.2 s pauses (streaming test).
    pause = np.zeros(int(1.2 * SR), np.float32)
    parts = []
    for sid, _ in SENTENCES:
        a, _ = load_audio(out / f"{sid}_{VOICES[0]}.wav", SR)
        parts += [a, pause]
    long_wav = save_wav(out / "long_ZuriNeural.wav", np.concatenate(parts), SR)
    manifest = {
        "name": out.name,
        "kind": "synthetic-tts",
        "warning": "Synthetic TTS speech; for integration tests only, not an accuracy benchmark.",
        "generator": "edge-tts", "voices": VOICES, "sample_rate": SR,
        "items": items,
        "long": {"audio": long_wav.name, "texts": [t for _, t in SENTENCES], "pause_s": 1.2},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} clips + long clip to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/sw_tts_v1")
    asyncio.run(main(Path(ap.parse_args().out)))
