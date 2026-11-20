"""Download a Hugging Face MT model and convert it once to CTranslate2.

The converted model lives in ``<cache>/ct2/<model>--<sha12>--<quant>/`` with a
``provenance.json`` recording the source repo, resolved commit, ctranslate2 version and
quantization. Conversion needs ``transformers`` and ``torch``; running the converted model
needs neither (only ``ctranslate2`` + the tokenizer).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .langs import tokenizer_files


class MtError(RuntimeError):
    pass


# Weights and tokenizer only: skip tf_model.h5, rust_model.ot, flax, onnx exports.
_ALLOW = ["*.json", "*.model", "*.spm", "*.txt", "pytorch_model*.bin", "model*.safetensors"]


def hf_snapshot(model_id: str, revision: str, cache_dir: Path, local_files_only: bool) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(model_id, revision=revision, cache_dir=str(cache_dir),
                                      allow_patterns=_ALLOW, local_files_only=local_files_only))
    except Exception as exc:
        raise MtError(f"could not fetch {model_id}@{revision}: {exc}") from exc
    return path, path.name  # snapshot dir name is the resolved commit sha


def stored_quantization(compute_type: str) -> str:
    return "int8" if compute_type.startswith("int8") else "float16"


def ct2_dir(cache_root: Path, model_id: str, sha: str, quant: str) -> Path:
    return cache_root / "ct2" / f"{model_id.replace('/', '--')}--{sha[:12]}--{quant}"


def find_converted(cache_root: Path, model_id: str, quant: str) -> Path | None:
    """Most recent converted copy of ``model_id`` (used when running offline)."""
    base = cache_root / "ct2"
    hits = sorted(base.glob(f"{model_id.replace('/', '--')}--*--{quant}"),
                  key=lambda p: p.stat().st_mtime, reverse=True) if base.exists() else []
    return next((p for p in hits if (p / "provenance.json").exists()), None)


def ensure_ct2_model(model_id: str, *, revision: str, family: str, compute_type: str,
                     cache_root: str | Path, local_files_only: bool = False) -> tuple[Path, dict]:
    cache_root = Path(cache_root)
    quant = stored_quantization(compute_type)
    t0 = time.perf_counter()
    try:
        snap, sha = hf_snapshot(model_id, revision, cache_root / "huggingface" / "hub", local_files_only)
    except MtError:
        existing = find_converted(cache_root, model_id, quant) if local_files_only else None
        if existing is None:
            raise
        info = json.loads((existing / "provenance.json").read_text(encoding="utf-8"))
        return existing, {**info, "fetch_ms": 0.0, "convert_ms": 0.0, "reused": True}
    fetch_ms = (time.perf_counter() - t0) * 1000
    out = ct2_dir(cache_root, model_id, sha, quant)
    prov_path = out / "provenance.json"
    if prov_path.exists():
        info = json.loads(prov_path.read_text(encoding="utf-8"))
        return out, {**info, "fetch_ms": round(fetch_ms, 1), "convert_ms": 0.0, "reused": True}

    try:
        import ctranslate2
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        raise MtError("ctranslate2 is not installed") from exc
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MtError("converting to CTranslate2 needs transformers and torch once "
                      "(pip install -e .[mt,torch]); or use mt.engine=transformers") from exc

    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    t1 = time.perf_counter()
    try:
        TransformersConverter(str(snap)).convert(str(tmp), quantization=quant, force=True)
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise MtError(f"CTranslate2 conversion of {model_id} failed: {exc}") from exc
    convert_ms = (time.perf_counter() - t1) * 1000
    tok_dir = tmp / "hf_tokenizer"   # kept apart: CTranslate2 writes its own config.json
    tok_dir.mkdir()
    for name in tokenizer_files(family):
        if (snap / name).exists():
            shutil.copy2(snap / name, tok_dir / name)
    info = {"model_id": model_id, "revision_requested": revision, "resolved_revision": sha,
            "family": family, "quantization": quant, "ctranslate2": ctranslate2.__version__,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (tmp / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out, {**info, "fetch_ms": round(fetch_ms, 1), "convert_ms": round(convert_ms, 1), "reused": False}
