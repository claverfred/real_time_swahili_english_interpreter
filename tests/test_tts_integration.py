"""Real TTS backends. Slow; skipped when the voice is not cached (no downloads in tests)."""

import sys

import numpy as np
import pytest

from src.config.schema import TtsConfig
from src.tts import Synthesizer, TtsError
from src.tts.backends import PiperBackend, SapiBackend

pytestmark = pytest.mark.slow
CFG = TtsConfig(model="x", voice="x")


def _loaded(backend):
    try:
        backend.load()
    except TtsError as exc:
        pytest.skip(f"backend unavailable: {exc}")
    return backend


@pytest.mark.parametrize("make", [
    lambda: PiperBackend("en_US-libritts_r-medium", revision="42301ec547bc2939386cf0ae9f9050ca05b1d3af",
                         allow_download=False),
    pytest.param(lambda: SapiBackend("default"),
                 marks=pytest.mark.skipif(sys.platform != "win32", reason="Windows SAPI")),
])
def test_real_backend_produces_consistent_speech(make):
    syn = Synthesizer([_loaded(make())], CFG)
    res = syn.synthesize("Good morning. The price is 2,500 shillings.")
    assert res.sample_rate == 22050 and len(res.chunks) == 2
    assert 1.5 < res.duration_s < 8.0
    rms = 20 * np.log10(np.sqrt(np.mean(res.audio ** 2)))
    assert -24 < rms < -16                                   # loudness-normalized near -20 dBFS
    assert all(c.warnings == [] for c in res.chunks)
    assert res.rtf < 1.0                                     # faster than real time on any recent CPU
