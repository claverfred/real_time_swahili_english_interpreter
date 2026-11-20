import asyncio
import time

import numpy as np
import pytest
import soundfile as sf

from src.audio import (
    AudioError,
    FrameChunker,
    analyze_levels,
    float_to_pcm16,
    load_audio,
    pcm16_to_float,
    resample,
    save_wav,
)
from src.audio.io import ensure_float32_mono
from src.capture import stream_array


def test_load_stereo_44k_int16_to_mono_16k(tmp_path):
    sr = 44100
    t = np.arange(sr * 2) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1) * 0.5
    path = tmp_path / "in.wav"
    sf.write(path, stereo, sr, subtype="PCM_16")
    audio, info = load_audio(path, 16000)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - 32000) <= 2
    assert (info.sample_rate_in, info.channels_in, info.decoder) == (44100, 2, "soundfile")
    assert info.duration_s == pytest.approx(2.0, abs=0.01)


def test_resample_preserves_frequency():
    sr_in, sr_out = 48000, 16000
    t = np.arange(sr_in) / sr_in
    y = resample(np.sin(2 * np.pi * 1000 * t).astype(np.float32), sr_in, sr_out)
    spectrum = np.abs(np.fft.rfft(y))
    peak_hz = np.argmax(spectrum) * sr_out / len(y)
    assert peak_hz == pytest.approx(1000, abs=2)


def test_pcm16_roundtrip():
    x = np.linspace(-1, 1, 1001, dtype=np.float32)
    assert np.max(np.abs(pcm16_to_float(float_to_pcm16(x)) - x)) < 1 / 32767 + 1e-6
    with pytest.raises(AudioError):
        pcm16_to_float(b"\x00")


def test_malformed_missing_empty_and_limits(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF this is not audio" * 10)
    with pytest.raises(AudioError):
        load_audio(bad)
    with pytest.raises(AudioError, match="not found"):
        load_audio(tmp_path / "missing.wav")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioError, match="empty"):
        load_audio(empty)
    ok = save_wav(tmp_path / "ok.wav", np.zeros(16000 * 3, dtype=np.float32), 16000)
    with pytest.raises(AudioError, match="max_input_seconds"):
        load_audio(ok, max_seconds=2)
    with pytest.raises(AudioError, match="max_input_mb"):
        load_audio(ok, max_mb=0.01)


def test_ensure_mono_validation():
    with pytest.raises(AudioError, match="NaN"):
        ensure_float32_mono(np.array([0.0, np.nan]))
    with pytest.raises(AudioError, match="empty"):
        ensure_float32_mono(np.array([]))
    assert ensure_float32_mono(np.array([32767, -32767], dtype=np.int16)).tolist() == pytest.approx([1.0, -1.0])
    assert ensure_float32_mono(np.ones((2, 100))).shape == (100,)   # channels-first


def test_level_warnings():
    sr = 16000
    assert "silent" in " ".join(analyze_levels(np.zeros(sr, dtype=np.float32), sr)["warnings"])
    loud = np.sign(np.sin(np.arange(sr))).astype(np.float32)
    assert any("clipping" in w for w in analyze_levels(loud, sr)["warnings"])
    quiet = (np.sin(np.arange(sr) / 5) * 0.006).astype(np.float32)   # ~-47 dBFS: quiet, not silent
    assert any("quiet" in w for w in analyze_levels(quiet, sr)["warnings"])


def test_frame_chunker_any_chunk_sizes():
    data = np.arange(1000, dtype=np.float32)
    ch = FrameChunker(160)
    frames = []
    for i in range(0, 1000, 37):
        frames += ch.push(data[i:i + 37])
    assert len(frames) == 6 and ch.pending_samples == 40
    assert np.array_equal(np.concatenate(frames), data[:960])
    last = ch.flush()
    assert len(last) == 160 and np.array_equal(last[:40], data[960:]) and not last[40:].any()
    assert ch.flush() is None


def test_stream_array_blocks_and_pacing():
    async def collect(factor):
        return [c async for c in stream_array(np.zeros(4800, dtype=np.float32), 16000, 100, factor)]

    assert [len(c) for c in asyncio.run(collect(0))] == [1600, 1600, 1600]
    t0 = time.perf_counter()
    asyncio.run(collect(1.0))
    assert time.perf_counter() - t0 >= 0.18   # 3 blocks of 100 ms -> waits ~200 ms
