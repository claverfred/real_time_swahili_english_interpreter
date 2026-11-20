import json
import subprocess
import sys

import numpy as np
import pytest

from src.audio import load_audio
from src.config.schema import VadConfig
from src.utils.manifest import FallbackLog
from src.vad import (
    EnergyVad,
    SileroVad,
    VadSegmenter,
    VadUnavailable,
    WebRtcVad,
    create_vad,
    segment_audio,
)
from src.vad.backends import SILERO_VERSION, ensure_silero_model

from .conftest import tone_bursts

CFG = VadConfig()


def test_energy_vad_finds_bursts_in_noise():
    segs = segment_audio(tone_bursts(), VadSegmenter(EnergyVad(), CFG))
    assert len(segs) == 2
    assert segs[0].start_s == pytest.approx(1.0 - 0.2, abs=0.1)   # includes speech_pad_ms
    assert segs[1].end_s == pytest.approx(3.2 + 0.3, abs=0.1)     # includes hangover_ms


@pytest.mark.parametrize("make", [lambda: WebRtcVad(0.5), EnergyVad])
def test_no_segments_in_silence_or_faint_noise(make):
    rng = np.random.default_rng(1)
    for x in (np.zeros(16000 * 3, np.float32), (rng.standard_normal(48000) * 0.0005).astype(np.float32)):
        assert segment_audio(x, VadSegmenter(make(), CFG)) == []


def test_webrtc_threshold_maps_to_aggressiveness():
    assert [WebRtcVad(t).aggressiveness for t in (0.0, 0.3, 0.5, 0.8, 1.0)] == [0, 1, 2, 3, 3]


def test_factory_falls_back_and_logs(tmp_path):
    cfg = VadConfig(backend="silero", silero_model_path=str(tmp_path / "missing.onnx"))
    fb = FallbackLog()
    backend = create_vad(cfg, cache_dir=str(tmp_path), fallbacks=fb)
    assert backend.name == "webrtc"
    assert fb.records[0].stage == "vad" and fb.records[0].selected == "webrtc"
    with pytest.raises(VadUnavailable):
        create_vad(cfg, cache_dir=str(tmp_path), allow_fallback=False)


def test_silero_checksum_is_enforced(tmp_path):
    fake = tmp_path / "models" / f"silero_vad_{SILERO_VERSION}" / "silero_vad.onnx"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"tampered")
    with pytest.raises(VadUnavailable, match="checksum"):
        ensure_silero_model(tmp_path, allow_download=False)
    with pytest.raises(VadUnavailable, match="allow_model_download"):
        ensure_silero_model(tmp_path / "empty", allow_download=False)


def test_silero_low_probability_on_silence(silero_path):
    vad = SileroVad(silero_path)
    probs = [vad(np.zeros(512, np.float32)) for _ in range(30)]
    assert max(probs) < 0.2


@pytest.mark.parametrize("backend", ["silero", "webrtc", "energy"])
def test_backends_agree_on_real_speech(backend, speech_wav, request):
    if backend == "silero":
        request.getfixturevalue("silero_path")
    audio, _ = load_audio(speech_wav)
    vad = create_vad(VadConfig(backend=backend), allow_fallback=False)
    segs = segment_audio(audio, VadSegmenter(vad, CFG))
    # "Good morning." | "How are you today?" | 1.5 s pause | "I am going to the market."
    assert len(segs) == 3
    assert segs[2].start_s - segs[1].end_s > 1.2
    assert all(s.reason == "silence" for s in segs)


def test_cli_vad_export(speech_wav, tmp_path):
    res = subprocess.run(
        [sys.executable, "-m", "src.cli", "vad", "--input", str(speech_wav), "--profile", "cpu",
         "--backend", "energy", "--export", "--save-segments",
         "--set", f"storage.output_dir={tmp_path.as_posix()}", "--set", f"storage.log_dir={tmp_path.as_posix()}"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    run_dir = next(tmp_path.glob("vad-*"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["vad_backend"] == "energy"
    assert len(json.loads((run_dir / "segments.json").read_text())) == 3
    assert len(list((run_dir / "segments").glob("*.wav"))) == 3
    assert (run_dir / "events.jsonl").is_file()


def test_cli_rejects_malformed_audio(tmp_path):
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"garbage" * 20)
    res = subprocess.run([sys.executable, "-m", "src.cli", "inspect", "--input", str(bad)],
                         capture_output=True, text=True)
    assert res.returncode == 3 and "error" in res.stderr
