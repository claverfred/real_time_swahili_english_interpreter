"""Use any Hugging Face Whisper checkpoint with faster-whisper.

* Repos that already hold a CTranslate2 model (``model.bin``, e.g. ``Systran/faster-whisper-*``,
  ``Sunbird/faster-whisper-*``) are used as downloaded.
* Transformers checkpoints (community fine-tunes such as Swahili Whisper models) are converted
  once to ``<cache>/ct2/asr/<model>--<sha12>--<quant>/`` with a ``provenance.json``.
  Conversion needs ``transformers`` + ``torch`` once; inference needs neither.

faster-whisper needs ``tokenizer.json`` and ``preprocessor_config.json``. Some fine-tunes ship
only the slow tokenizer files, or omit the preprocessor config. The tokenizer is rebuilt from the
repo's own files, and the feature extractor settings are written from ``config.json``
(``num_mel_bins``: 80 for large-v2 and earlier, 128 for large-v3 / turbo).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

# Short names faster-whisper resolves itself (Systran CTranslate2 repos).
BUILTIN = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "distil-small.en", "medium", "medium.en",
           "distil-medium.en", "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-large-v3",
           "large-v3-turbo", "turbo", "distil-large-v3.5"}
_META = ["*.json", "*.txt", "*.model", "vocabulary.*"]


class AsrConvertError(RuntimeError):
    pass


def is_builtin(model: str) -> bool:
    return model in BUILTIN or Path(model).is_dir()


def ct2_asr_dir(cache_root: Path, model_id: str, sha: str, quant: str) -> Path:
    return cache_root / "ct2" / "asr" / f"{model_id.replace('/', '--')}--{sha[:12]}--{quant}"


def find_converted_asr(cache_root: Path, model_id: str, quant: str) -> Path | None:
    base = cache_root / "ct2" / "asr"
    hits = sorted(base.glob(f"{model_id.replace('/', '--')}--*--{quant}"),
                  key=lambda p: p.stat().st_mtime, reverse=True) if base.exists() else []
    return next((p for p in hits if (p / "provenance.json").exists()), None)


def _weights_patterns(model_id: str, revision: str) -> list[str]:
    """Prefer CTranslate2 weights, then safetensors, then PyTorch .bin (never download two formats)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(model_id, revision=revision)
    if "model.bin" in files:
        return ["model.bin"]
    if any(f.endswith(".safetensors") for f in files):
        return ["model*.safetensors"]
    if any(f.startswith("pytorch_model") and f.endswith(".bin") for f in files):
        return ["pytorch_model*.bin"]
    raise AsrConvertError(f"{model_id}: no Whisper weights found (files: {files[:20]})")


def _snapshot(model_id: str, revision: str, cache_dir: Path, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    patterns = _META + (["*"] if local_files_only else _weights_patterns(model_id, revision))
    try:
        return Path(snapshot_download(model_id, revision=revision, cache_dir=str(cache_dir),
                                      allow_patterns=patterns, local_files_only=local_files_only))
    except OSError as exc:
        # Windows without Developer Mode: the hub cache links files with identical content
        # (e.g. added_tokens.json) and fails with WinError 1314. Fall back to a plain-copy folder
        # named after the resolved commit, so provenance stays exact.
        if getattr(exc, "winerror", None) != 1314 or local_files_only:
            raise
        from huggingface_hub import HfApi

        sha = HfApi().model_info(model_id, revision=revision).sha
        target = cache_dir.parent / "local" / model_id.replace("/", "--") / sha
        snapshot_download(model_id, revision=sha, local_dir=str(target), allow_patterns=patterns)
        return target


def _write_support_files(snap: Path, out: Path) -> None:
    if (snap / "tokenizer.json").exists():
        shutil.copy2(snap / "tokenizer.json", out / "tokenizer.json")
    else:
        from transformers import WhisperTokenizerFast

        tmp = out / "_tok"
        WhisperTokenizerFast.from_pretrained(str(snap)).save_pretrained(str(tmp))
        shutil.move(str(tmp / "tokenizer.json"), str(out / "tokenizer.json"))
        shutil.rmtree(tmp, ignore_errors=True)
    if (snap / "preprocessor_config.json").exists():
        shutil.copy2(snap / "preprocessor_config.json", out / "preprocessor_config.json")
    else:
        mels = json.loads((snap / "config.json").read_text(encoding="utf-8")).get("num_mel_bins", 80)
        (out / "preprocessor_config.json").write_text(json.dumps({
            "chunk_length": 30, "feature_size": mels, "hop_length": 160, "n_fft": 400, "n_samples": 480000,
            "nb_max_frames": 3000, "padding_value": 0.0, "sampling_rate": 16000}, indent=2), encoding="utf-8")


def _compatible_source(snap: Path, work: Path) -> Path:
    """Snapshot view the installed transformers can read. Checkpoints saved by transformers 5
    store ``extra_special_tokens`` as a list, which 4.x rejects; it only names tokens that are
    already in the vocabulary, so it is dropped. Files are hard-linked (no copy, cache untouched)."""
    cfg_path = snap / "tokenizer_config.json"
    if not cfg_path.exists():
        return snap
    tok_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(tok_cfg.get("extra_special_tokens"), list):
        return snap
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in snap.iterdir():
        if f.name == "tokenizer_config.json" or not f.is_file():
            continue
        real = f.resolve()
        try:
            (work / f.name).hardlink_to(real)
        except OSError:
            shutil.copy2(real, work / f.name)
    tok_cfg.pop("extra_special_tokens")
    (work / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")
    return work


def ensure_ct2_whisper(model_id: str, *, revision: str = "main", compute_type: str = "int8",
                       cache_root: str | Path = "cache", local_files_only: bool = False) -> tuple[Path, dict]:
    """Return a directory faster-whisper can load, converting the checkpoint once if needed."""
    from huggingface_hub import constants

    cache_root = Path(cache_root)
    quant = "int8" if compute_type.startswith("int8") else "float16"
    local_files_only = local_files_only or bool(constants.HF_HUB_OFFLINE)   # honour HF_HUB_OFFLINE=1
    t0 = time.perf_counter()
    try:
        snap = _snapshot(model_id, revision, cache_root / "huggingface" / "hub", local_files_only)
    except AsrConvertError:
        raise
    except Exception as exc:
        existing = find_converted_asr(cache_root, model_id, quant) if local_files_only else None
        if existing is None:
            raise AsrConvertError(f"could not fetch {model_id}@{revision}: {exc}") from exc
        info = json.loads((existing / "provenance.json").read_text(encoding="utf-8"))
        return existing, {**info, "reused": True}
    sha = snap.name
    fetch_ms = round((time.perf_counter() - t0) * 1000, 1)
    if (snap / "model.bin").exists():          # already CTranslate2
        return snap, {"model_id": model_id, "resolved_revision": sha, "format": "ctranslate2", "fetch_ms": fetch_ms}

    out = ct2_asr_dir(cache_root, model_id, sha, quant)
    if (out / "provenance.json").exists():
        info = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        return out, {**info, "fetch_ms": fetch_ms, "reused": True}
    try:
        import ctranslate2
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        raise AsrConvertError("ctranslate2 is not installed") from exc
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise AsrConvertError(f"{model_id} is a transformers checkpoint; converting it needs transformers and "
                              "torch once (pip install -e .[mt,torch])") from exc
    tmp = out.with_name(out.name + ".partial")
    src = out.with_name(out.name + ".src")
    shutil.rmtree(tmp, ignore_errors=True)
    t1 = time.perf_counter()
    try:
        src_dir = _compatible_source(snap, src)
        TransformersConverter(str(src_dir)).convert(str(tmp), quantization=quant, force=True)
        _write_support_files(src_dir, tmp)
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise AsrConvertError(f"CTranslate2 conversion of {model_id} failed: {exc}") from exc
    finally:
        shutil.rmtree(src, ignore_errors=True)
    info = {"model_id": model_id, "revision_requested": revision, "resolved_revision": sha, "format": "converted",
            "quantization": quant, "ctranslate2": ctranslate2.__version__,
            "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "convert_ms": round((time.perf_counter() - t1) * 1000, 1)}
    (tmp / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out, {**info, "fetch_ms": fetch_ms, "reused": False}
