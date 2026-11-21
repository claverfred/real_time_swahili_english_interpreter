"""TTS backends. Each returns float32 mono audio at its native rate; ``Synthesizer`` makes the
output consistent (rate, loudness, trimming, fades).

| backend | runs   | license (engine / voice data)                    |
|---------|--------|--------------------------------------------------|
| piper   | local  | GPL-3.0 engine (piper1-gpl) / per-voice MODEL_CARD |
| kokoro  | local  | MIT wrapper, Apache-2.0 model (hexgrad/Kokoro-82M) |
| sapi    | local  | Windows built-in voices (Windows license)        |
| mms     | local  | CC-BY-NC-4.0 (facebook/mms-tts-eng)               |
| edge    | cloud  | Microsoft online service terms; text is sent to Microsoft |
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np


class TtsError(RuntimeError):
    pass


class TtsBackend:
    name = "base"
    is_cloud = False

    def __init__(self, voice: str, *, cache_root: str | Path = "cache", speaker_id: int | None = None,
                 speaking_rate: float = 1.0, allow_download: bool = True, revision: str = "main",
                 timeout_s: float = 30.0) -> None:
        self.voice, self.speaker_id, self.speaking_rate = voice, speaker_id, speaking_rate
        self.cache_root, self.allow_download, self.revision = Path(cache_root), allow_download, revision
        self.timeout_s = timeout_s
        self.sample_rate = 0
        self.license = "unknown"
        self.load_info: dict[str, object] = {}

    def load(self) -> dict[str, object]:
        raise NotImplementedError

    def synthesize(self, text: str) -> np.ndarray:
        raise NotImplementedError


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- Piper ------------------------------------------------------------------------------------

def piper_voice_path(voice: str) -> str:
    """'en_US-joe-medium' -> 'en/en_US/joe/medium/en_US-joe-medium'."""
    try:
        locale, name, quality = voice.split("-")
    except ValueError as exc:
        raise TtsError(f"piper voice must look like en_US-joe-medium, got {voice!r}") from exc
    return f"{locale.split('_')[0]}/{locale}/{name}/{quality}/{voice}"


class PiperBackend(TtsBackend):
    name = "piper"
    REPO = "rhasspy/piper-voices"

    def load(self) -> dict[str, object]:
        try:
            from huggingface_hub import hf_hub_download
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:
            raise TtsError("piper is not installed (pip install -e .[tts])") from exc
        base = piper_voice_path(self.voice)
        t0 = time.perf_counter()
        try:  # hf_hub_download verifies LFS files against their sha256
            files = {ext: Path(hf_hub_download(self.REPO, f"{base}{ext}", revision=self.revision,
                                               cache_dir=str(self.cache_root / "huggingface" / "hub"),
                                               local_files_only=not self.allow_download))
                     for ext in (".onnx", ".onnx.json")}
            card = Path(hf_hub_download(self.REPO, f"{base.rsplit('/', 1)[0]}/MODEL_CARD", revision=self.revision,
                                        cache_dir=str(self.cache_root / "huggingface" / "hub"),
                                        local_files_only=not self.allow_download))
        except Exception as exc:
            raise TtsError(f"could not fetch piper voice {self.voice}: {exc}") from exc
        fetch_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        try:
            self._voice = PiperVoice.load(files[".onnx"], config_path=files[".onnx.json"])
        except Exception as exc:
            raise TtsError(f"piper could not load {self.voice}: {exc}") from exc
        self._syn = SynthesisConfig(speaker_id=self.speaker_id, length_scale=1.0 / self.speaking_rate)
        self.sample_rate = int(self._voice.config.sample_rate)
        card_text = card.read_text(encoding="utf-8")
        lic = next((ln.split(":", 1)[1].strip() for ln in card_text.splitlines() if "License:" in ln), "see MODEL_CARD")
        self.license = f"voice data: {lic}; engine: GPL-3.0"
        self.load_info = {"backend": self.name, "voice": self.voice, "revision": self.revision,
                          "model_sha256": _sha256(files[".onnx"]), "sample_rate": self.sample_rate,
                          "license": self.license, "fetch_ms": round(fetch_ms, 1),
                          "load_ms": round((time.perf_counter() - t1) * 1000, 1)}
        return self.load_info

    def synthesize(self, text: str) -> np.ndarray:
        parts = [c.audio_float_array for c in self._voice.synthesize(text, syn_config=self._syn)]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


# -- Kokoro -----------------------------------------------------------------------------------

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
# SHA-256 measured on download 2026-11-20 (88.1 MB / 26.9 MB); a mismatch is refused.
KOKORO_FILES: dict[str, str | None] = {
    "kokoro-v1.0.int8.onnx": "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}


class KokoroBackend(TtsBackend):
    name = "kokoro"

    def _fetch(self, fname: str) -> Path:
        d = self.cache_root / "models" / "kokoro-v1.0"
        path = d / fname
        if not path.exists():
            if not self.allow_download:
                raise TtsError(f"{path} missing and tts.allow_model_download is false")
            d.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"{KOKORO_RELEASE}/{fname}", tmp)  # noqa: S310 - fixed https URL
            tmp.replace(path)
        digest = _sha256(path)
        if KOKORO_FILES.get(fname) and digest != KOKORO_FILES[fname]:
            raise TtsError(f"{path} SHA-256 mismatch ({digest}); delete it and retry")
        return path

    def load(self) -> dict[str, object]:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise TtsError("kokoro-onnx is not installed (pip install kokoro-onnx)") from exc
        t0 = time.perf_counter()
        try:
            model, voices = (self._fetch(f) for f in KOKORO_FILES)
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(f"could not fetch Kokoro model: {exc}") from exc
        fetch_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        try:
            self._k = Kokoro(str(model), str(voices))
        except Exception as exc:
            raise TtsError(f"Kokoro could not load: {exc}") from exc
        if self.voice not in self._k.get_voices():
            raise TtsError(f"unknown Kokoro voice {self.voice!r}")
        self.sample_rate = 24000
        self.license = "model Apache-2.0 (hexgrad/Kokoro-82M); kokoro-onnx MIT; espeak-ng GPL-3.0"
        self.load_info = {"backend": self.name, "voice": self.voice, "sample_rate": self.sample_rate,
                          "license": self.license, "model_sha256": KOKORO_FILES["kokoro-v1.0.int8.onnx"],
                          "fetch_ms": round(fetch_ms, 1), "load_ms": round((time.perf_counter() - t1) * 1000, 1)}
        return self.load_info

    def synthesize(self, text: str) -> np.ndarray:
        lang = "en-gb" if self.voice[:1] == "b" else "en-us"
        audio, sr = self._k.create(text, voice=self.voice, speed=self.speaking_rate, lang=lang)
        self.sample_rate = sr
        return np.asarray(audio, dtype=np.float32)


# -- Windows SAPI -----------------------------------------------------------------------------

class SapiBackend(TtsBackend):
    """Built-in Windows voices through COM. No download; offline. One COM object per thread."""
    name = "sapi"
    _FORMAT_22K_16BIT_MONO = 22

    def _thread_voice(self):
        local = self.__dict__.setdefault("_local", threading.local())
        if getattr(local, "voice", None) is None:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            v = comtypes.client.CreateObject("SAPI.SpVoice")
            if self.voice not in ("", "default"):
                tokens = v.GetVoices()
                match = [tokens.Item(i) for i in range(tokens.Count)
                         if self.voice.lower() in tokens.Item(i).GetDescription().lower()]
                if not match:
                    raise TtsError(f"no SAPI voice matching {self.voice!r}")
                v.Voice = match[0]
            v.Rate = int(max(-10, min(10, round((self.speaking_rate - 1.0) * 10))))
            local.voice, local.client = v, comtypes.client
        return local.voice, local.client

    def load(self) -> dict[str, object]:
        t0 = time.perf_counter()
        try:
            v, _ = self._thread_voice()
            desc = v.Voice.GetDescription()
        except ImportError as exc:
            raise TtsError("SAPI needs Windows and comtypes (pip install comtypes)") from exc
        except TtsError:
            raise
        except Exception as exc:
            raise TtsError(f"SAPI unavailable: {exc}") from exc
        self.sample_rate = 22050
        self.license = "Windows built-in voice (Windows license terms)"
        self.load_info = {"backend": self.name, "voice": desc, "sample_rate": self.sample_rate,
                          "license": self.license, "load_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return self.load_info

    def synthesize(self, text: str) -> np.ndarray:
        v, client = self._thread_voice()
        stream = client.CreateObject("SAPI.SpMemoryStream")
        fmt = client.CreateObject("SAPI.SpAudioFormat")
        fmt.Type = self._FORMAT_22K_16BIT_MONO
        stream.Format = fmt
        v.AudioOutputStream = stream
        v.Speak(text, 0)
        data = bytes(stream.GetData())
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


# -- MMS (VITS via transformers) --------------------------------------------------------------

class MmsBackend(TtsBackend):
    name = "mms"

    def load(self) -> dict[str, object]:
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel
        except ImportError as exc:
            raise TtsError("mms needs transformers and torch") from exc
        t0 = time.perf_counter()
        try:
            kw = {"revision": self.revision, "cache_dir": str(self.cache_root / "huggingface" / "hub"),
                  "local_files_only": not self.allow_download}
            self._model = VitsModel.from_pretrained(self.voice, **kw).eval()
            self._tok = AutoTokenizer.from_pretrained(self.voice, **kw)
        except Exception as exc:
            raise TtsError(f"could not load {self.voice}: {exc}") from exc
        self._torch = torch
        self.sample_rate = int(self._model.config.sampling_rate)
        self._model.config.speaking_rate = self.speaking_rate
        self.license = "CC-BY-NC-4.0 (non-commercial)"
        self.load_info = {"backend": self.name, "voice": self.voice, "sample_rate": self.sample_rate,
                          "license": self.license, "load_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return self.load_info

    def synthesize(self, text: str) -> np.ndarray:
        enc = self._tok(text, return_tensors="pt")
        with self._torch.inference_mode():
            wav = self._model(**enc).waveform[0]
        return wav.numpy().astype(np.float32)


# -- Edge (cloud) -----------------------------------------------------------------------------

class EdgeBackend(TtsBackend):
    """Microsoft Edge online voices (edge-tts). Sends the English text to Microsoft over the
    internet; only allowed when fallback.allow_cloud is set and the data class permits it."""
    name = "edge"
    is_cloud = True

    def load(self) -> dict[str, object]:
        try:
            import edge_tts  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError as exc:
            raise TtsError("edge needs edge-tts and soundfile") from exc
        self.sample_rate = 24000
        self.license = "Microsoft online service terms (unofficial edge-tts client)"
        t0 = time.perf_counter()
        self.synthesize("Ready.")          # proves network + voice now, not mid-session
        self.load_info = {"backend": self.name, "voice": self.voice, "sample_rate": self.sample_rate,
                          "license": self.license, "cloud": True,
                          "probe_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return self.load_info

    async def _stream(self, text: str) -> bytes:
        import edge_tts
        pct = int(round((self.speaking_rate - 1.0) * 100))
        comm = edge_tts.Communicate(text, self.voice, rate=f"{pct:+d}%",
                                    connect_timeout=int(self.timeout_s), receive_timeout=int(self.timeout_s))
        buf = bytearray()
        async for msg in comm.stream():
            if msg["type"] == "audio":
                buf += msg["data"]
        return bytes(buf)

    def synthesize(self, text: str) -> np.ndarray:
        import soundfile as sf
        try:
            mp3 = asyncio.run(asyncio.wait_for(self._stream(text), self.timeout_s))
        except Exception as exc:
            raise TtsError(f"edge-tts request failed: {type(exc).__name__}: {exc}") from exc
        if not mp3:
            raise TtsError("edge-tts returned no audio")
        audio, sr = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=False)
        self.sample_rate = sr
        return audio if audio.ndim == 1 else audio.mean(axis=1)


BACKENDS = {"piper": PiperBackend, "kokoro": KokoroBackend, "sapi": SapiBackend, "mms": MmsBackend,
            "edge": EdgeBackend}


def make_backend(name: str, voice: str, **kwargs) -> TtsBackend:
    try:
        cls = BACKENDS[name]
    except KeyError as exc:
        raise TtsError(f"unknown TTS backend {name!r}; choose from {sorted(BACKENDS)}") from exc
    return cls(voice, **kwargs)
