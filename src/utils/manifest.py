"""Run manifests and fallback records. Every run is reproducible from its manifest.

Baselines are never overwritten: each run gets a fresh, unique directory.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__

TRACKED_PACKAGES = [
    "pydantic", "PyYAML", "psutil", "numpy", "torch", "torchaudio", "faster-whisper",
    "ctranslate2", "transformers", "sentencepiece", "edge-tts", "piper-tts", "soundfile",
    "soxr", "sounddevice", "webrtcvad-wheels", "fastapi", "gradio", "jiwer", "sacrebleu",
]


@dataclass
class FallbackRecord:
    stage: str           # device | precision | asr | mt | tts | vad
    requested: str
    selected: str
    reason: str
    wall_time: float = field(default_factory=time.time)


class FallbackLog:
    """Collects fallbacks so none can happen silently."""

    def __init__(self) -> None:
        self.records: list[FallbackRecord] = []

    def add(self, stage: str, requested: str, selected: str, reason: str) -> FallbackRecord:
        rec = FallbackRecord(stage, requested, selected, reason)
        self.records.append(rec)
        return rec

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]


def package_versions(names: list[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def git_commit(repo: Path | None = None) -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=repo, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               timeout=5, cwd=repo).stdout.strip()
        return res.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.SubprocessError, OSError):
        return None


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def new_run_dir(output_dir: Path, prefix: str = "run") -> Path:
    """Create a run directory that is guaranteed not to exist yet."""
    while True:
        d = output_dir / new_run_id(prefix)
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue


def build_manifest(
    *,
    run_id: str,
    mode: str,
    config: dict[str, Any],
    runtime: dict[str, Any],
    fallbacks: FallbackLog,
    datasets: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software_version": __version__,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "packages": package_versions(),
        "runtime": runtime,
        "config": config,
        "fallbacks": fallbacks.to_list(),
        "datasets": datasets or [],
        "extra": extra or {},
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest {path}")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
