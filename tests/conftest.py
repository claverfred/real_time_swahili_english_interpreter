import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.vad.backends import SILERO_VERSION

SILERO_CACHED = Path("cache") / "models" / f"silero_vad_{SILERO_VERSION}" / "silero_vad.onnx"

_SAPI = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($args[0], $f)
$p = New-Object System.Speech.Synthesis.PromptBuilder
$p.AppendText("Good morning. How are you today?")
$p.AppendBreak([TimeSpan]::FromSeconds(1.5))
$p.AppendText("I am going to the market.")
$s.Speak($p); $s.Dispose()
"""


@pytest.fixture(scope="session")
def speech_wav(tmp_path_factory) -> Path:
    """Real (synthesized) speech: 3 utterances, the last after a 1.5 s pause. Windows only."""
    if sys.platform != "win32":
        pytest.skip("speech fixture uses Windows SAPI")
    out = tmp_path_factory.mktemp("speech") / "speech.wav"
    script = out.with_suffix(".ps1")
    script.write_text(_SAPI, encoding="utf-8")
    res = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(out)],
                         capture_output=True, text=True, timeout=60)
    if res.returncode != 0 or not out.exists():
        pytest.skip(f"SAPI synthesis unavailable: {res.stderr[:200]}")
    return out


@pytest.fixture
def silero_path() -> Path:
    if not SILERO_CACHED.is_file():
        pytest.skip("Silero model not cached (run: python -m src.cli vad --backend silero --input <wav>)")
    return SILERO_CACHED


def tone_bursts(sr=16000, total_s=4.0, bursts=((1.0, 1.5), (2.5, 3.2)), noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(int(total_s * sr)) * noise).astype(np.float32)
    for a, b in bursts:
        t = np.arange(int((b - a) * sr)) / sr
        x[int(a * sr):int(a * sr) + len(t)] += 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    return x
