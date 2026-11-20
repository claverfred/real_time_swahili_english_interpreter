"""Path resolution and storage checks. No machine-specific absolute paths in code."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ..config.schema import InterpreterConfig


def resolve_path(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(os.fspath(p))).resolve()


def prepare_storage(cfg: InterpreterConfig) -> dict[str, Path]:
    """Create cache/output/log dirs and point model caches at the configured cache dir."""
    paths = {
        "cache_dir": resolve_path(cfg.storage.cache_dir),
        "output_dir": resolve_path(cfg.storage.output_dir),
        "log_dir": resolve_path(cfg.storage.log_dir),
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    # Respect a user-provided HF_HOME; otherwise keep model downloads inside the cache dir.
    os.environ.setdefault("HF_HOME", str(paths["cache_dir"] / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(paths["cache_dir"] / "torch"))
    return paths


def free_disk_gb(path: str | os.PathLike[str]) -> float:
    p = resolve_path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return round(shutil.disk_usage(p).free / 1e9, 2)
