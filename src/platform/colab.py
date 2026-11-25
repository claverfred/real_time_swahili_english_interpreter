"""Thin Google Colab helpers. The notebooks call these; the pipeline itself is the shared code.

Nothing here hard-codes a session path: the persistent root is Google Drive when it is mounted
(``MyDrive/<root_name>``) and ``/content/<root_name>`` otherwise, and it is published to the
config through ``SWI_COLAB_ROOT`` (see ``configs/colab.yaml``).

Every function also runs outside Colab (it returns ``False`` / empty values or raises a clear
error), so it can be unit-tested on any machine.
"""

from __future__ import annotations

import base64
import ctypes
import glob
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from .detect import is_colab
from .secrets import get_secret

ROOT_ENV = "SWI_COLAB_ROOT"
DRIVE_MOUNT = "/content/drive"
# Packages Colab ships and the notebook kernel may already have imported. Installing our extras
# must never replace them, or the runtime needs a restart (and torch/CUDA builds could mismatch).
PROTECTED = ("torch", "torchaudio", "torchvision", "numpy", "scipy", "jax", "jaxlib", "tensorflow")
# Directories kept on the persistent root and linked into the repository checkout.
PERSISTENT_DIRS = ("data/fleurs_sw", "data/flores200", "data/raw", "data/synthetic")
REPORT_PACKAGES = ("torch", "ctranslate2", "faster-whisper", "transformers", "onnxruntime", "onnxruntime-gpu",
                   "piper-tts", "sacrebleu", "jiwer", "huggingface-hub", "numpy", "fastapi", "uvicorn")


# -- storage ---------------------------------------------------------------------------------------
def mount_drive(mount_point: str = DRIVE_MOUNT) -> bool:
    """Mount Google Drive (asks the user for permission in the browser). False outside Colab."""
    if not is_colab():
        return False
    if Path(mount_point, "MyDrive").is_dir():
        return True
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point)
    return Path(mount_point, "MyDrive").is_dir()


def persistent_root(root_name: str = "swahili_interpreter", *, use_drive: bool = True,
                    drive_mount: str = DRIVE_MOUNT, local_base: str = "/content") -> tuple[Path, bool]:
    """(root, on_drive). Drive survives runtime resets; the local disk is wiped with the VM."""
    if use_drive and Path(drive_mount, "MyDrive").is_dir():
        return Path(drive_mount, "MyDrive", root_name), True
    return Path(local_base, root_name), False


def setup_storage(repo: str | os.PathLike[str] = ".", *, root_name: str = "swahili_interpreter",
                  use_drive: bool = True, environ: dict[str, str] | None = None,
                  drive_mount: str = DRIVE_MOUNT, local_base: str = "/content") -> dict[str, object]:
    """Create the persistent layout, export ``SWI_COLAB_ROOT`` / ``HF_HOME``, and link the
    dataset folders of the checkout to the persistent root. Returns what was done."""
    environ = os.environ if environ is None else environ
    root, on_drive = persistent_root(root_name, use_drive=use_drive, drive_mount=drive_mount,
                                     local_base=local_base)
    for sub in ("cache", "results", "logs", "env", "inputs", *PERSISTENT_DIRS):
        (root / sub).mkdir(parents=True, exist_ok=True)
    environ[ROOT_ENV] = str(root)
    environ.setdefault("HF_HOME", str(root / "cache" / "huggingface"))
    environ.setdefault("SWI_PROFILE", "colab")
    links = link_persistent_dirs(repo, root)
    notes = [] if on_drive else ["Google Drive not mounted: models, data and results are lost when the runtime is reset"]
    return {"root": str(root), "on_drive": on_drive, "links": links, "notes": notes}


def link_persistent_dirs(repo: str | os.PathLike[str], root: str | os.PathLike[str],
                         dirs: Iterable[str] = PERSISTENT_DIRS) -> list[str]:
    """Replace ``<repo>/<dir>`` by a symlink to ``<root>/<dir>``. Existing files in the checkout
    folder are moved to the persistent copy first (nothing is deleted)."""
    repo, root = Path(repo), Path(root)
    probe = repo / ".swi-symlink-probe"
    try:                                  # e.g. Windows without Developer Mode cannot symlink
        probe.unlink(missing_ok=True)
        probe.symlink_to(root, target_is_directory=True)
        probe.unlink()
    except OSError as exc:
        return [f"not linked ({exc.__class__.__name__}: symlinks unavailable); data stays in the checkout"]
    done = []
    for d in dirs:
        target, link = root / d, repo / d
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if Path(os.readlink(link)) == target:
                continue
            link.unlink()
        elif link.is_dir():
            for child in list(link.iterdir()):
                dest = target / child.name
                if not dest.exists():
                    shutil.move(str(child), str(dest))
            if any(link.iterdir()):           # name clashes: keep the checkout's copies aside
                link.rename(link.with_name(f"{link.name}.local-{time.strftime('%Y%m%d-%H%M%S')}"))
            else:
                link.rmdir()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        done.append(f"{d} -> {target}")
    return done


# -- credentials -----------------------------------------------------------------------------------
def hf_login_from_secrets() -> bool:
    """Log in to Hugging Face with the HF_TOKEN Colab Secret, without printing it.
    Not required: every default model is public. A token only raises download rate limits."""
    token = get_secret("HF_TOKEN")
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except ImportError:
        pass
    return True


# -- CUDA libraries --------------------------------------------------------------------------------
def cuda_library_dirs(site_dirs: Iterable[str] | None = None) -> list[str]:
    """``lib`` folders of the pip CUDA packages (nvidia-cudnn-cu12, nvidia-cublas-cu12, ...)
    that torch installs. CTranslate2 needs cuDNN 9 / cuBLAS 12 but does not look there itself."""
    if site_dirs is None:
        import site

        site_dirs = [*site.getsitepackages(), site.getusersitepackages()]
    out: list[str] = []
    for s in site_dirs:
        for d in sorted(glob.glob(os.path.join(s, "nvidia", "*", "lib"))):
            if d not in out:
                out.append(d)
    return out


def export_cuda_library_path(environ: dict[str, str] | None = None,
                             site_dirs: Iterable[str] | None = None) -> list[str]:
    """Prepend those folders to ``LD_LIBRARY_PATH``: affects ``!python ...`` subprocesses."""
    environ = os.environ if environ is None else environ
    dirs = cuda_library_dirs(site_dirs)
    current = [p for p in environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    new = [d for d in dirs if d not in current]
    if new:
        environ["LD_LIBRARY_PATH"] = os.pathsep.join(new + current)
    return new


def preload_cuda_libraries(prefixes: Iterable[str] = ("libcublas", "libcublasLt", "libcudnn")) -> list[str]:
    """For in-kernel use (``LD_LIBRARY_PATH`` is read only at process start): load the pip
    CUDA libraries globally so CTranslate2's later ``dlopen`` by name finds them."""
    loaded = []
    if not sys.platform.startswith("linux"):
        return loaded
    prefixes = tuple(prefixes)
    for d in cuda_library_dirs():
        for f in sorted(os.listdir(d)):
            if f.startswith(prefixes) and ".so" in f:
                try:
                    ctypes.CDLL(os.path.join(d, f), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(f)
                except OSError:
                    pass
    return loaded


# -- environment report ----------------------------------------------------------------------------
def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        return subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
    except (subprocess.SubprocessError, OSError):
        return None


def environment_report() -> dict[str, object]:
    """Python, packages, GPU and CUDA libraries, for the run log and the provenance export."""
    from .cuda import probe_cuda

    cuda = probe_cuda()
    rep: dict[str, object] = {
        "colab": is_colab(), "colab_release": os.environ.get("COLAB_RELEASE_TAG"),
        "python": sys.version.split()[0], "platform": sys.platform,
        "packages": {p: _version(p) for p in REPORT_PACKAGES},
        "cuda": cuda.to_dict(), "nvidia_smi": nvidia_smi(), "cuda_library_dirs": cuda_library_dirs(),
        "persistent_root": os.environ.get(ROOT_ENV), "hf_token_set": bool(get_secret("HF_TOKEN")),
    }
    try:
        import torch

        rep["torch_cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    except Exception:
        rep["torch_cudnn"] = None
    try:
        import ctranslate2

        rep["ctranslate2_cuda_devices"] = ctranslate2.get_cuda_device_count()
        rep["ctranslate2_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cuda")) \
            if rep["ctranslate2_cuda_devices"] else []
    except Exception as exc:
        rep["ctranslate2_cuda_devices"] = f"error: {exc}"
    return rep


def write_lock_file(out_dir: str | os.PathLike[str]) -> Path:
    """``pip freeze`` of this runtime (the package lock for the run's provenance)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pip-freeze-{time.strftime('%Y%m%d-%H%M%S')}.txt"
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    path.write_text(freeze, encoding="utf-8")
    return path


def protected_constraints(packages: Iterable[str] = PROTECTED) -> list[str]:
    """``name==version`` for the preinstalled packages the install must keep."""
    return [f"{p}=={v}" for p in packages if (v := _version(p))]


def restart_required(modules: Mapping[str, object] | None = None) -> list[str]:
    """Packages whose imported version differs from the installed one (restart the runtime)."""
    modules = sys.modules if modules is None else modules
    stale = []
    for mod, dist in (("numpy", "numpy"), ("torch", "torch"), ("transformers", "transformers"),
                      ("ctranslate2", "ctranslate2"), ("huggingface_hub", "huggingface-hub")):
        m = modules.get(mod)
        loaded, installed = getattr(m, "__version__", None), _version(dist)
        if m is not None and loaded and installed and loaded != installed:
            stale.append(f"{dist}: imported {loaded}, installed {installed}")
    return stale


# -- GPU memory ------------------------------------------------------------------------------------
def gpu_memory() -> dict[str, float] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {"used_mb": round((total - free) / 2**20), "total_mb": round(total / 2**20)}


def free_gpu_memory(*objects: object) -> dict[str, float] | None:
    """Drop references the caller passes (models, interpreters), collect, empty the CUDA cache.
    CTranslate2 frees its memory when the model object is garbage-collected."""
    import gc

    for o in objects:
        close = getattr(o, "close", None) or getattr(o, "unload", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return gpu_memory()


# -- export ----------------------------------------------------------------------------------------
def zip_runs(run_dirs: Iterable[str | os.PathLike[str]], out_zip: str | os.PathLike[str],
             extra_files: Iterable[str | os.PathLike[str]] = ()) -> dict[str, object]:
    """Zip run directories (JSON, CSV, audio, logs, images, manifests) into one archive."""
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    n, size = 0, 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for d in map(Path, run_dirs):
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                z.write(f, Path(d.name) / f.relative_to(d))
                n, size = n + 1, size + f.stat().st_size
        for f in map(Path, extra_files):
            if f.is_file():
                z.write(f, f.name)
                n, size = n + 1, size + f.stat().st_size
    return {"zip": str(out_zip), "files": n, "uncompressed_mb": round(size / 2**20, 2),
            "zip_mb": round(out_zip.stat().st_size / 2**20, 2)}


def latest_runs(results_dir: str | os.PathLike[str], prefix: str = "", n: int = 1) -> list[Path]:
    d = Path(results_dir)
    runs = sorted((p for p in d.glob(f"{prefix}*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return runs[-n:] if n else runs


def download(path: str | os.PathLike[str]) -> bool:
    """Offer a file to the browser (Colab only)."""
    if not is_colab():
        return False
    from google.colab import files  # type: ignore[import-not-found]

    files.download(str(path))
    return True


# -- audio input -----------------------------------------------------------------------------------
def upload_audio(dest: str | os.PathLike[str]) -> list[Path]:
    """Ask the user to pick audio files in the browser; saved under ``dest``."""
    if not is_colab():
        raise RuntimeError("upload_audio works only in Google Colab; pass a file path instead")
    from google.colab import files  # type: ignore[import-not-found]

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, data in files.upload().items():
        p = dest / Path(name).name            # never trust client paths
        p.write_bytes(data)
        saved.append(p)
    return saved


_RECORD_JS = """
async function swiRecord(maxSeconds) {
  const stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true}});
  const rec = new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  const box = document.createElement('div');
  box.style = 'font:14px sans-serif;padding:8px;border:1px solid #c33;border-radius:6px;display:inline-block';
  const btn = document.createElement('button');
  btn.textContent = 'Stop recording';
  const label = document.createElement('span');
  label.style = 'margin-left:10px';
  box.append(btn, label);
  document.body.append(box);
  const t0 = Date.now();
  const timer = setInterval(() => { label.textContent = 'Recording… ' + ((Date.now() - t0) / 1000).toFixed(0) + ' s'; }, 250);
  const done = new Promise(resolve => rec.onstop = resolve);
  rec.start();
  const stop = () => { if (rec.state !== 'inactive') rec.stop(); };
  btn.onclick = stop;
  setTimeout(stop, maxSeconds * 1000);
  await done;
  clearInterval(timer);
  stream.getTracks().forEach(t => t.stop());
  box.remove();
  const buf = await new Blob(chunks).arrayBuffer();
  let bin = '';
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}
"""


def record_browser_audio(dest: str | os.PathLike[str], max_seconds: float = 30.0,
                         sample_rate: int = 16000) -> Path:
    """Record from the user's browser microphone (the browser asks for permission; a Stop button
    ends it early). Returns a 16 kHz mono WAV. The recording stays in this runtime/Drive folder."""
    if not is_colab():
        raise RuntimeError("browser recording works only in Google Colab; use `--mic` locally")
    from google.colab import output  # type: ignore[import-not-found]
    from IPython.display import Javascript, display  # type: ignore[import-not-found]

    display(Javascript(_RECORD_JS))
    b64 = output.eval_js(f"swiRecord({float(max_seconds)})", timeout_sec=max_seconds + 60)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw = dest / f"mic-{stamp}.webm"
    raw.write_bytes(base64.b64decode(b64))
    wav = dest / f"mic-{stamp}.wav"
    decode_to_wav(raw, wav, sample_rate)
    raw.unlink(missing_ok=True)
    return wav


def decode_to_wav(src: str | os.PathLike[str], wav: str | os.PathLike[str], sample_rate: int = 16000) -> Path:
    """Browser recordings are WebM/Opus; ffmpeg (preinstalled on Colab) converts them."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is needed to decode browser recordings")
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(sample_rate),
                    str(wav)], check=True)
    return Path(wav)


def proxy_url(port: int) -> str | None:
    """Public HTTPS URL of a port in this runtime, served through Colab's authenticated proxy."""
    if not is_colab():
        return None
    from google.colab import output  # type: ignore[import-not-found]

    return str(output.eval_js(f"google.colab.kernel.proxyPort({int(port)})")).rstrip("/")


def results_dir() -> Path:
    """Where the ``colab`` profile writes run directories."""
    root = os.environ.get(ROOT_ENV)
    return Path(root, "results") if root else Path("results")


class GpuMonitor:
    """Sample GPU memory with nvidia-smi in a background thread (covers subprocesses, unlike
    torch's own counters). ``peak_mb`` is None when there is no NVIDIA GPU."""

    def __init__(self, index: int = 0, interval_s: float = 0.5) -> None:
        import threading

        self.index, self.interval_s = index, interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._exe = shutil.which("nvidia-smi")

    def _sample(self) -> int | None:
        try:
            out = subprocess.run([self._exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                  "-i", str(self.index)], capture_output=True, text=True, timeout=5).stdout
            return int(float(out.strip().splitlines()[0]))
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            return None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            v = self._sample()
            if v is not None:
                self.samples.append(v)

    def __enter__(self) -> GpuMonitor:
        if self._exe:
            v = self._sample()
            self.baseline_mb = v
            self._thread.start()
        else:
            self.baseline_mb = None
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def peak_mb(self) -> int | None:
        return max(self.samples) if self.samples else None


def run_cli(args: list[str], *, prefix: str | None = None, echo: bool = True,
            gpu_index: int = 0) -> dict[str, object]:
    """Run ``python -m src.cli <args>`` in a subprocess (GPU memory is released when it exits),
    streaming its output, and return the new run directory and the peak GPU memory."""
    out_dir = results_dir()
    before = {p.name for p in out_dir.glob("*")} if out_dir.is_dir() else set()
    t0 = time.perf_counter()
    lines: list[str] = []
    with GpuMonitor(gpu_index) as mon:
        proc = subprocess.Popen([sys.executable, "-m", "src.cli", *args], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if echo:
                print(line, end="", flush=True)
        rc = proc.wait()
    new = [p for p in out_dir.glob(f"{prefix or ''}*") if p.is_dir() and p.name not in before] if out_dir.is_dir() else []
    run_dir = max(new, key=lambda p: p.stat().st_mtime) if new else None
    errors = [ln.strip() for ln in lines if ln.lstrip().lower().startswith("error:")]
    return {"returncode": rc, "run_dir": str(run_dir) if run_dir and rc == 0 else None,
            "seconds": round(time.perf_counter() - t0, 1), "gpu_peak_mb": mon.peak_mb,
            "gpu_baseline_mb": mon.baseline_mb, "tail": "".join(lines[-15:]),
            "error": errors[-1] if errors else (None if rc == 0 else lines[-1].strip() if lines else None)}


NO_GPU = ("No GPU in this Colab runtime. The '{profile}' profile runs on CUDA and benchmarks refuse to "
          "fall back to CPU. Fix: Runtime > Change runtime type > T4 GPU (or better) > Save, then run all "
          "cells again. To run on CPU on purpose (slow; same models as the development PC), set PROFILE = \"cpu\".")


def require_gpu(profile: str = "colab", status: object = None) -> str:
    """Stop a notebook early, with an actionable message, when the profile needs CUDA but the
    runtime has none. Returns the GPU name (or "cpu" for a profile that does not need one)."""
    from ..config import load_config
    from .cuda import probe_cuda

    st = status if status is not None else probe_cuda()
    if st.available and st.gpus:                                   # type: ignore[attr-defined]
        return str(st.gpus[0].name)                                # type: ignore[attr-defined]
    if load_config(profile).environment.device == "cuda":
        raise RuntimeError(NO_GPU.format(profile=profile))
    return "cpu"


def bootstrap(repo: str | os.PathLike[str] = ".", *, use_drive: bool = True,
              root_name: str = "swahili_interpreter") -> dict[str, object]:
    """Notebook kernel setup, safe to re-run (also after a runtime restart): mount Drive, set up
    persistent storage, Hugging Face login from Colab Secrets, CUDA library path for CTranslate2."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    drive = mount_drive() if use_drive else False
    info = setup_storage(repo, root_name=root_name, use_drive=use_drive and drive)
    info["hf_login"] = hf_login_from_secrets()
    info["cuda_library_dirs_added"] = export_cuda_library_path()
    return info


def save_json(obj: object, path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return p
