"""Environment diagnosis: Python, packages, FFmpeg, audio devices, CUDA, RAM, disk, secrets."""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Literal

from .cpu import cpu_info
from .cuda import probe_cuda
from .detect import detect_platform
from .hpc import is_login_node, slurm_info
from .secrets import has_secret
from .storage import free_disk_gb

Status = Literal["ok", "warn", "fail", "info"]

# import name -> (distribution name, extras group, why it matters)
PACKAGES: dict[str, tuple[str, str, str]] = {
    "pydantic": ("pydantic", "core", "configuration"),
    "yaml": ("PyYAML", "core", "configuration"),
    "psutil": ("psutil", "core", "resource monitoring"),
    "numpy": ("numpy", "core", "audio arrays"),
    "soundfile": ("soundfile", "audio", "WAV/FLAC I/O"),
    "soxr": ("soxr", "audio", "resampling"),
    "webrtcvad": ("webrtcvad-wheels", "audio", "WebRTC VAD"),
    "sounddevice": ("sounddevice", "local", "microphone capture / playback"),
    "faster_whisper": ("faster-whisper", "asr", "Whisper ASR (CTranslate2)"),
    "ctranslate2": ("ctranslate2", "asr", "INT8 inference"),
    "transformers": ("transformers", "mt", "NLLB / Marian MT"),
    "sentencepiece": ("sentencepiece", "mt", "MT tokenization"),
    "torch": ("torch", "torch", "Silero VAD, MT on GPU"),
    "edge_tts": ("edge-tts", "tts", "Edge TTS (cloud)"),
    "piper": ("piper-tts", "tts", "Piper TTS (offline)"),
    "fastapi": ("fastapi", "ui", "local API / WebSocket"),
    "gradio": ("gradio", "ui", "browser UI"),
    "jiwer": ("jiwer", "eval", "WER / CER"),
    "sacrebleu": ("sacrebleu", "eval", "BLEU / chrF"),
}


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    hint: str = ""


def _pkg_checks() -> list[Check]:
    checks = []
    for mod, (dist, group, why) in PACKAGES.items():
        if importlib.util.find_spec(mod) is None:
            status: Status = "fail" if group == "core" else "warn"
            checks.append(Check(f"package:{dist}", status, f"missing ({why})",
                                f'pip install -e ".[{group}]"' if group != "core" else "pip install -e ."))
        else:
            try:
                ver = md.version(dist)
            except md.PackageNotFoundError:
                ver = "installed"
            checks.append(Check(f"package:{dist}", "ok", ver))
    return checks


def _audio_device_check() -> Check:
    if importlib.util.find_spec("sounddevice") is None:
        return Check("audio_devices", "warn", "sounddevice not installed; microphone unavailable",
                     'pip install -e ".[local]"')
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        n_in = sum(1 for d in devices if d["max_input_channels"] > 0)
        n_out = sum(1 for d in devices if d["max_output_channels"] > 0)
        status: Status = "ok" if n_in and n_out else "warn"
        return Check("audio_devices", status, f"{n_in} input / {n_out} output devices",
                     "" if n_in else "No microphone found; check OS privacy settings and drivers")
    except Exception as exc:  # PortAudio missing or driver error
        return Check("audio_devices", "fail", f"PortAudio error: {exc}",
                     "Install PortAudio (Linux: apt install libportaudio2) or check audio drivers")


def run_diagnosis(cache_dir: str = "cache") -> dict[str, object]:
    checks: list[Check] = []

    py = sys.version_info
    if py < (3, 10):
        checks.append(Check("python", "fail", platform.python_version(), "Python >= 3.10 required"))
    elif py >= (3, 13):
        checks.append(Check("python", "warn", platform.python_version(),
                            "Some ML wheels (ctranslate2, torch) may lag new Python releases; 3.10-3.12 recommended"))
    else:
        checks.append(Check("python", "ok", platform.python_version()))

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    checks.append(Check("virtualenv", "ok" if in_venv else "warn",
                        sys.prefix if in_venv else "not in a virtual environment",
                        "" if in_venv else "Create one: py -3.10 -m venv .venv"))

    checks.extend(_pkg_checks())

    ffmpeg = shutil.which("ffmpeg")
    checks.append(Check("ffmpeg", "ok" if ffmpeg else "warn", ffmpeg or "not on PATH",
                        "" if ffmpeg else "Needed for MP3/M4A input. Windows: winget install Gyan.FFmpeg"))

    checks.append(_audio_device_check())

    cuda = probe_cuda()
    checks.append(Check("cuda", "ok" if cuda.available else "info",
                        ", ".join(f"{g.name} ({g.memory_total_mb} MB)" for g in cuda.gpus)
                        if cuda.available else "; ".join(cuda.notes),
                        "" if cuda.available else "CPU profile will be used (--profile cpu)"))

    cpu = cpu_info()
    ram = float(cpu["ram_total_gb"])  # type: ignore[arg-type]
    checks.append(Check("ram", "ok" if ram >= 8 else "warn",
                        f"{cpu['ram_available_gb']} GB free of {ram} GB",
                        "" if ram >= 8 else "Under 8 GB RAM: use ASR model 'base' or 'small'"))

    disk = free_disk_gb(cache_dir)
    checks.append(Check("disk", "ok" if disk >= 10 else "warn", f"{disk} GB free at cache dir",
                        "" if disk >= 10 else "Model caches need ~10 GB; free space or move SWI_CACHE_DIR"))

    checks.append(Check("secret:HF_TOKEN", "ok" if has_secret("HF_TOKEN") else "info",
                        "present" if has_secret("HF_TOKEN") else "not set (only needed for gated models)"))

    plat = detect_platform()
    if plat == "hpc" and is_login_node():
        checks.append(Check("hpc_node", "warn", "login node",
                            "Run heavy work through Slurm (scripts/hpc/), not on the login node"))

    summary = {s: sum(1 for c in checks if c.status == s) for s in ("ok", "warn", "fail", "info")}
    return {
        "platform": plat,
        "os": platform.platform(),
        "cpu": cpu,
        "cuda": cuda.to_dict(),
        "slurm": slurm_info(),
        "checks": [asdict(c) for c in checks],
        "summary": summary,
    }
