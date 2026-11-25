"""Generate the Google Colab notebooks in notebooks/ (the .ipynb files are build outputs).

    python scripts/make_notebooks.py

Notebooks stay thin: every cell calls committed code (``src/``, ``scripts/``, the CLI). Edit the
cell sources here, regenerate, and commit both. Outputs are never stored in the notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
GITHUB = "claverfred/real_time_swahili_english_interpreter"
BADGE = ("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
         f"(https://colab.research.google.com/github/{GITHUB}/blob/main/notebooks/{{name}})")


GPU_CHECKS = {
    # Benchmarks stop here instead of failing every run later with "device: cuda unavailable".
    "require": '''
GPU = colab.require_gpu(PROFILE).replace(" ", "_")
print("GPU:", GPU)
''',
    "warn": '''
try:
    GPU = colab.require_gpu(PROFILE).replace(" ", "_")
    print("GPU:", GPU)
except RuntimeError as exc:
    GPU = "cpu"
    print("WARNING:", exc, "This notebook still runs on CPU, much more slowly.")
''',
    "none": "",
}


def setup_cell(setup_args: str, gpu: str = "require") -> str:
    return f'''# @title 1. Setup: clone/update the repository, mount Drive, install, fetch data (safe to re-run)
REPO_URL = "https://github.com/{GITHUB}.git"  # @param {{type:"string"}}
BRANCH = "main"  # @param {{type:"string"}}
USE_DRIVE = True  # @param {{type:"boolean"}}
PROFILE = "colab"  # @param {{type:"string"}}

import os, subprocess, sys
WORKDIR = "/content" if os.path.isdir("/content") else os.getcwd()
REPO = os.path.join(WORKDIR, REPO_URL.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git"))
if not os.path.isdir(os.path.join(REPO, ".git")):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
print(subprocess.run(["git", "log", "-1", "--format=commit %h  %cd  %s"], capture_output=True, text=True).stdout)

from src.platform import colab
info = colab.bootstrap(REPO, use_drive=USE_DRIVE)
print("persistent root:", info["root"], "(Google Drive)" if info["on_drive"] else "")
for n in info["notes"]:
    print("NOTE:", n)
print("Hugging Face token:", "from Colab Secrets" if info["hf_login"] else "none (fine: all default models are public)")

r = subprocess.run([sys.executable, "scripts/setup_colab.py", "--profile", PROFILE{setup_args}])
stale = colab.restart_required()
if stale:
    print("Restart the runtime (Runtime > Restart session), then run this cell again:", stale)
assert r.returncode == 0, "setup reported a failure (see the table above)"
''' + GPU_CHECKS[gpu]


EXPORT = '''# @title Export: ZIP this notebook's runs (JSON, CSV, audio, logs, images, manifests) and download
import time
zip_path = colab.results_dir() / f"{EXPORT_NAME}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
env_files = sorted((colab.results_dir().parent / "env").glob("*"))[-4:]
z = colab.zip_runs([p for p in RUN_DIRS if p], zip_path, extra_files=EXTRA_FILES + env_files)
print(z)
colab.download(zip_path)
'''

FREE = '''# @title Free GPU memory
print(colab.free_gpu_memory())
'''

TABLE_HELPERS = '''import pandas as pd
from IPython.display import display
from src.eval.summary import summarize_run

def table(runs, extra_keys=("label", "gpu_peak_mb", "seconds")):
    rows = []
    for r in runs:
        if not r.get("run_dir"):
            rows.append({"label": r.get("label"), "status": f"FAILED (exit {r['returncode']})",
                         "error": r.get("error")})
            continue
        for row in summarize_run(r["run_dir"]):
            rows.append({**{k: r.get(k) for k in extra_keys}, **row, "status": "ok"})
    df = pd.DataFrame(rows, columns=None if rows else ["label", "status"])
    if "status" in df and not (df.status == "ok").any():
        print("No candidate finished. The first error was:\\n ", next((r.get("error") for r in runs if r.get("error")), "?"))
    return df
'''

NOTHING_TO_PLOT = '''ok = df[df.status == "ok"].copy() if "status" in df else df.iloc[0:0]
if ok.empty:
    print("Nothing to plot: no candidate finished (see the status/error columns above).")
else:
'''


def indent(src: str) -> str:
    return "".join("    " + ln if ln.strip() else ln for ln in src.splitlines(keepends=True))


def nb(name: str, cells: list[tuple[str, str]]) -> dict:
    def cell(kind: str, src: str) -> dict:
        lines = src.strip("\n").splitlines(keepends=True)
        if kind == "md":
            return {"cell_type": "markdown", "metadata": {}, "source": lines}
        return {"cell_type": "code", "metadata": {"cellView": "form"} if src.lstrip().startswith("# @title") else {},
                "execution_count": None, "outputs": [], "source": lines}
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "name": name},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}},
            "cells": [cell(k, s) for k, s in cells]}


def header(name: str, title: str, body: str) -> tuple[str, str]:
    return ("md", f"# {title}\n\n{BADGE.format(name=name)}\n\n{body.strip()}\n\n"
                  "**Before you start:** Runtime → Change runtime type → **GPU** (T4 is enough; L4/A100/L40S are faster). "
                  "Optional: add a Hugging Face token as the Colab Secret `HF_TOKEN` (key icon on the left); "
                  "every default model is public, so it is not required.\n\n"
                  "Results, converted models and data are kept on Google Drive in `MyDrive/swahili_interpreter/`, "
                  "so later sessions start faster. Nothing is uploaded anywhere else.")


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {}

# --------------------------------------------------------------------------------------------------
NOTEBOOKS["01_environment_setup.ipynb"] = [
    header("01_environment_setup.ipynb", "01 · Environment setup and smoke test", """
Prepares a fresh Colab runtime and checks every stage on the GPU:

1. clone (or update) the repository, mount Google Drive, Hugging Face login from Colab Secrets;
2. install the pinned extras **without replacing Colab's torch/numpy** (no restart needed);
3. download the evaluation data once (FLEURS Swahili test, FLORES-200; kept on Drive);
4. record the environment (GPU, CUDA, cuDNN, package versions, `pip freeze` lock file);
5. smoke test: config → MT → TTS → ASR on a real recording → full pipeline.

The first run downloads and converts the models (about 5-10 minutes). Later runs reuse them from Drive.
"""),
    ("code", setup_cell("", gpu="warn")),
    ("code", '''# @title 2. GPU and package report
import json
rep = colab.environment_report()
print(rep["nvidia_smi"] or "nvidia-smi not found: no GPU attached (Runtime > Change runtime type)")
print(json.dumps({k: rep[k] for k in ("python", "packages", "torch_cudnn", "ctranslate2_cuda_devices",
                                      "ctranslate2_compute_types", "cuda_library_dirs")}, indent=2))
print("GPU memory:", colab.gpu_memory())
'''),
    ("code", '''# @title 3. Resolved configuration (device, precision, models) and diagnosis
!python -m src.cli config show --profile {PROFILE} --resolve | head -n 60
!python -m src.cli diagnose
'''),
    ("code", '''# @title 4. Dependency smoke test: the fast unit tests (no model downloads, ~1 minute)
!python -m pytest -q -m "not slow and not gpu" -p no:cacheprovider 2>&1 | tail -n 5
'''),
    ("md", "If every row above says `ok`, the other notebooks will run. A `warn` row explains itself "
           "(e.g. running on CPU). Re-running this notebook later only updates the repository and repeats the checks."),
    ("code", '''# @title 5. Export the environment report and lock file
RUN_DIRS, EXTRA_FILES, EXPORT_NAME = [], [], "environment"
''' + EXPORT.split("\n", 1)[1]),
]

# --------------------------------------------------------------------------------------------------
NOTEBOOKS["02_asr_benchmark.ipynb"] = [
    header("02_asr_benchmark.ipynb", "02 · Swahili ASR benchmark on GPU", """
Compares Swahili speech-recognition candidates on **real read speech** (FLEURS `sw_ke` test, 487
recordings, human transcripts) through the deployed pipeline (VAD → ASR → correction list).
Each candidate runs in its own process with `fallback.policy=fail`, so a model that cannot load
is reported as failed instead of being silently replaced. GPU memory is sampled with `nvidia-smi`.

Reported per candidate: WER with 95% bootstrap CI, WER with number words as digits, CER, empty
outputs, decode time p50/p95/p99, real-time factor, peak GPU memory.
The CPU reference (development PC, 10 cores, all 487 recordings): Mollel/ASR-Swahili-Small, beam 5, WER 0.272.
"""),
    ("code", setup_cell(', "--no-smoke"')),
    ("code", '''# @title 2. Candidates
N = 487  # @param {type:"integer"}
CANDIDATES = [  # (label, model, revision, beam)
    ("Mollel small, beam 5 (cpu/colab default)", "Mollel/ASR-Swahili-Small", "b2bab6c17b39a0d9b17549444535d4018782a958", 5),
    ("Mollel small, beam 1", "Mollel/ASR-Swahili-Small", "b2bab6c17b39a0d9b17549444535d4018782a958", 1),
    ("cdli small afrivoice, beam 5", "cdli/whisper-small-swahili-afrivoice_cv", "main", 5),
    ("OpenAI large-v3-turbo, beam 5", "large-v3-turbo", "main", 5),
    ("OpenAI large-v3, beam 5", "large-v3", "main", 5),
]
'''),
    ("code", '''# @title 3. Run (one process per candidate)
runs = []
for label, model, rev, beam in CANDIDATES:
    print(f"\\n===== {label} =====")
    r = colab.run_cli(["evaluate", "speech", "--n", str(N), "--no-oracle", "--bootstrap", "500", "--profile", PROFILE,
                       "--set", f"asr.model={model}", "--set", f"asr.revision={rev}", "--set", f"asr.beam_size={beam}",
                       "--set", "fallback.policy=fail"], prefix="eval-speech")
    runs.append({"label": label, **r})
    print(f"-> exit {r['returncode']}, {r['seconds']} s, peak GPU memory {r['gpu_peak_mb']} MB"
          + (f"\\n-> FAILED: {r['error']}" if r["returncode"] else ""))
'''),
    ("code", '''# @title 4. Comparison table (saved as CSV next to the runs)
''' + TABLE_HELPERS + '''
df = table(runs)
cols = ["label", "status", "error", "n", "wer", "wer_ci95", "wer_numbers_as_digits", "cer", "empty_rate",
        "decode_p50_s", "decode_p95_s", "decode_p99_s", "rtf", "gpu_peak_mb", "device", "asr_model"]
display(df[[c for c in cols if c in df]])
csv = colab.results_dir() / f"colab_asr_comparison_{GPU}.csv"
df.to_csv(csv, index=False)
print("saved", csv)
'''),
    ("code", '''# @title 5. Chart: WER with 95% CI
import matplotlib.pyplot as plt
png = None
''' + NOTHING_TO_PLOT + indent('''lo_hi = ok.wer_ci95.str.split("-", expand=True).astype(float)
fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(ok)))
ax.barh(ok.label, ok.wer, xerr=[ok.wer - lo_hi[0], lo_hi[1] - ok.wer], color="#2a78d6", height=0.5, capsize=3)
ax.invert_yaxis(); ax.set_xlabel("WER (lower is better)"); ax.set_title(f"FLEURS sw_ke test, n={N}, {GPU}")
ax.grid(axis="x", color="#e1e0d9"); ax.set_axisbelow(True)
png = colab.results_dir() / f"colab_asr_comparison_{GPU}.png"
fig.savefig(png, dpi=150, bbox_inches="tight"); plt.show()
''')),
    ("code", '''RUN_DIRS, EXTRA_FILES, EXPORT_NAME = [r["run_dir"] for r in runs], [f for f in (csv, png) if f], "asr-benchmark"
''' + EXPORT),
]

# --------------------------------------------------------------------------------------------------
NOTEBOOKS["03_mt_benchmark.ipynb"] = [
    header("03_mt_benchmark.ipynb", "03 · Swahili → English translation benchmark on GPU", """
Compares translation candidates on **FLORES-200 devtest** (1012 sentences, independent professional
human translations, never model output). Each candidate runs in its own process with `fallback.policy=fail`.

Reported: BLEU / chrF / chrF++ with 95% bootstrap CIs (sacrebleu signatures in each report), digits kept,
names vs reference, omission/addition proxies, latency p50/p95/p99 per sentence, peak GPU memory.

Licenses matter: every NLLB checkpoint is **CC-BY-NC-4.0 (non-commercial)**; Marian opus-mt is Apache-2.0.
The CPU reference (development PC): NLLB-600M int8, beam 2, BLEU 39.0, chrF 61.6.
"""),
    ("code", setup_cell(', "--no-smoke"')),
    ("code", '''# @title 2. Candidates
N = 1012  # @param {type:"integer"}
CANDIDATES = [  # (label, backend, model, beam)
    ("NLLB-600M distilled, beam 4 (colab default)", "nllb", "facebook/nllb-200-distilled-600M", 4),
    ("NLLB-600M distilled, beam 2 (cpu default)", "nllb", "facebook/nllb-200-distilled-600M", 2),
    ("NLLB-1.3B distilled, beam 4", "nllb", "facebook/nllb-200-distilled-1.3B", 4),
    ("NLLB-1.3B, beam 4", "nllb", "facebook/nllb-200-1.3B", 4),
    ("NLLB-3.3B, beam 4", "nllb", "facebook/nllb-200-3.3B", 4),
    ("Marian opus-mt-swc-en, beam 4", "marian", "Helsinki-NLP/opus-mt-swc-en", 4),
]
'''),
    ("code", '''# @title 3. Run (one process per candidate)
runs = []
for label, backend, model, beam in CANDIDATES:
    print(f"\\n===== {label} =====")
    r = colab.run_cli(["evaluate", "mt", "--n", str(N), "--profile", PROFILE, "--set", f"mt.backend={backend}",
                       "--set", f"mt.model={model}", "--set", "mt.revision=main", "--set", f"mt.beam_size={beam}",
                       "--set", "fallback.policy=fail"], prefix="eval-mt")
    runs.append({"label": label, **r})
    print(f"-> exit {r['returncode']}, {r['seconds']} s, peak GPU memory {r['gpu_peak_mb']} MB"
          + (f"\\n-> FAILED: {r['error']}" if r["returncode"] else ""))
'''),
    ("code", '''# @title 4. Comparison table (saved as CSV)
''' + TABLE_HELPERS + '''
df = table(runs)
cols = ["label", "status", "error", "n", "bleu", "bleu_ci95", "chrf", "chrf_ci95", "chrf_pp", "numbers_kept", "names_vs_ref",
        "omission_rate", "latency_p50_s", "latency_p95_s", "latency_p99_s", "gpu_peak_mb", "mt_model"]
display(df[[c for c in cols if c in df]])
csv = colab.results_dir() / f"colab_mt_comparison_{GPU}.csv"
df.to_csv(csv, index=False)
print("saved", csv)
'''),
    ("code", '''# @title 5. Chart: quality vs latency
import matplotlib.pyplot as plt
png = None
''' + NOTHING_TO_PLOT + indent('''fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(ok.latency_p50_s, ok.chrf, s=60, color="#2a78d6")
for _, r in ok.iterrows():
    ax.annotate(r.label, (r.latency_p50_s, r.chrf), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("median latency per sentence (s)"); ax.set_ylabel("chrF (higher is better)")
ax.set_title(f"FLORES-200 devtest, n={N}, {GPU}"); ax.grid(color="#e1e0d9")
png = colab.results_dir() / f"colab_mt_comparison_{GPU}.png"
fig.savefig(png, dpi=150, bbox_inches="tight"); plt.show()
''')),
    ("code", '''RUN_DIRS, EXTRA_FILES, EXPORT_NAME = [r["run_dir"] for r in runs], [f for f in (csv, png) if f], "mt-benchmark"
''' + EXPORT),
]

# --------------------------------------------------------------------------------------------------
NOTEBOOKS["04_tts_benchmark.ipynb"] = [
    header("04_tts_benchmark.ipynb", "04 · English TTS benchmark", """
Compares English voices on first-audio latency, real-time factor, memory and an **intelligibility proxy**
(each clip is transcribed back with Whisper and scored by WER against the input text). Text: 25 FLORES-200
English sentences plus a fixed set of names, numbers and acronyms.

The proxy is not a listening test: naturalness needs human ratings (`docs/human_evaluation.md`).
Piper runs on the CPU through onnxruntime even on a GPU runtime; MMS uses PyTorch (GPU when available).
The CPU reference (development PC): Piper libritts_r first audio 0.22 s median, RTF 0.06, round-trip WER 0.074.
"""),
    ("code", setup_cell(', "--no-smoke"', gpu="warn")),
    ("code", '''# @title 2. Candidates (backend:voice)
INSTALL_KOKORO = False  # @param {type:"boolean"}
CANDIDATES = ["piper:en_US-libritts_r-medium", "piper:en_US-joe-medium", "piper:en_US-lessac-medium",
              "mms:facebook/mms-tts-eng"]
if INSTALL_KOKORO:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kokoro-onnx>=0.6"], check=True)
    CANDIDATES.append("kokoro:af_heart")
N_SENTENCES = 25  # @param {type:"integer"}
'''),
    ("code", '''# @title 3. Run
import time
before = {p.name for p in colab.results_dir().glob("tts-eval*")}
t0 = time.perf_counter()
with colab.GpuMonitor() as mon:
    r = subprocess.run([sys.executable, "scripts/tts_eval.py", "--profile", PROFILE, "--n", str(N_SENTENCES),
                        "--candidates", *CANDIDATES])
new = sorted(p for p in colab.results_dir().glob("tts-eval*") if p.name not in before)
run_dir = new[-1] if new else None
print("exit", r.returncode, "| run", run_dir, "| peak GPU MB", mon.peak_mb, "|", round(time.perf_counter() - t0), "s")
'''),
    ("code", '''# @title 4. Table (saved as CSV)
import json, pandas as pd
from IPython.display import display
rows = []
for x in json.loads((run_dir / "results.json").read_text(encoding="utf-8")):
    if x["status"] != "ok":
        rows.append({"candidate": x["candidate"], "status": "FAILED: " + x.get("error", "")[:80]}); continue
    rows.append({"candidate": x["candidate"], "status": "ok", "license": x["license"],
                 "first_audio_p50_s": round(x["first_audio_ms"]["median"] / 1000, 3),
                 "first_audio_p95_s": round(x["first_audio_ms"]["p95"] / 1000, 3), "rtf": x["rtf"],
                 "roundtrip_wer": x["intelligibility_all"]["wer"],
                 "roundtrip_wer_names_numbers": x["intelligibility_names_numbers"]["wer"],
                 "clips_with_warnings": x["clips_with_warnings"], "rss_delta_mb": x["rss_delta_mb"]})
df = pd.DataFrame(rows); display(df)
csv = colab.results_dir() / f"colab_tts_comparison_{GPU}.csv"; df.to_csv(csv, index=False); print("saved", csv)
'''),
    ("code", '''# @title 5. Listen: the same sentence in each voice
from IPython.display import Audio
TEXT = "President Samia Suluhu Hassan arrived in Dodoma on Tuesday; maize prices rose by 12 percent."  # @param {type:"string"}
samples = []
for c in CANDIDATES:
    backend, voice = c.split(":", 1)
    wav = run_dir / f"sample_{backend}_{voice.replace('/', '_')}.wav"
    subprocess.run([sys.executable, "-m", "src.cli", "speak", "--profile", PROFILE, "--backend", backend, "--voice", voice,
                    "--text", TEXT, "--output", str(wav)], capture_output=True)
    if wav.exists():
        print(c); display(Audio(str(wav))); samples.append(wav)
    else:
        print(c, "- synthesis failed")
'''),
    ("code", '''RUN_DIRS, EXTRA_FILES, EXPORT_NAME = [run_dir], [csv], "tts-benchmark"
''' + EXPORT),
]

# --------------------------------------------------------------------------------------------------
NOTEBOOKS["05_live_demo.ipynb"] = [
    header("05_live_demo.ipynb", "05 · Live demo: speak Swahili, hear English", """
Loads the whole pipeline once on the GPU (VAD → Swahili ASR → stabilization → MT → English TTS) and keeps it
in memory, then interprets:

* **your browser microphone** (turn-taking: speak, press *Stop* or wait, then hear the English);
* an **uploaded** audio file, or a file on **Google Drive**;
* a FLEURS recording streamed at **real-time pace** (listener latency, as in a live session).

**Privacy:** recordings are saved only in this runtime and, if Drive is mounted, in
`MyDrive/swahili_interpreter/inputs/` (delete them with the last cell). Nothing is sent to any other service:
every model runs inside this Colab runtime.
"""),
    ("code", setup_cell(', "--no-smoke"', gpu="warn")),
    ("code", '''# @title 2. Load the interpreter (models stay on the GPU until the last cell)
colab.preload_cuda_libraries()
from IPython.display import Audio, display
from src.pipeline import InterpreterSession

session = InterpreterSession(PROFILE)
m = session.model_status()
print(f"loaded in {session.load_s} s on {session.interp.runtime.device}: ASR {m['asr']['model']} | MT {m['mt']['model']} | "
      f"TTS {m['tts']['backend']} ({m['tts']['voice']})")
print("GPU memory:", colab.gpu_memory())

def show(r):
    print("Swahili:", r["swahili"] or "(no speech recognised)")
    print("English:", r["english"] or "-")
    for u in r["units"]:
        first = u.get("lat_first_audio_ms")
        print(f"   unit: ASR {u.get('lat_asr_final_ms')} ms, MT {u.get('lat_mt_final_ms')} ms, "
              f"first audio {first} ms after the speaker stopped")
    print(f"input {r['input_s']} s, processed in {r['wall_s']} s" + (" (real-time pace)" if r["realtime"] else ""))
    for e in r["errors"]:
        print("ERROR:", e)
    if len(r["audio"]):
        display(Audio(r["audio"], rate=r["sample_rate"], autoplay=True))
'''),
    ("code", '''# @title 3. Browser microphone: run, allow the microphone, speak Swahili, press Stop
MAX_SECONDS = 20  # @param {type:"integer"}
INPUTS = os.path.join(info["root"], "inputs")
wav = colab.record_browser_audio(INPUTS, max_seconds=MAX_SECONDS)
print("recorded", wav)
show(session.interpret(wav))
'''),
    ("code", '''# @title 4. Upload an audio file (wav, mp3, m4a, ogg, flac, webm)
for p in colab.upload_audio(os.path.join(info["root"], "inputs")):
    src = p if p.suffix.lower() in (".wav", ".flac", ".ogg") else colab.decode_to_wav(p, p.with_suffix(".wav"))
    print("==", p.name); show(session.interpret(src))
'''),
    ("code", '''# @title 5. A file on Google Drive
DRIVE_FILE = "/content/drive/MyDrive/path/to/swahili.wav"  # @param {type:"string"}
show(session.interpret(DRIVE_FILE))
'''),
    ("code", '''# @title 6. A FLEURS recording streamed at real-time pace (listener latency)
import json
ITEM = 400  # @param {type:"integer"}
lines = open("data/fleurs_sw/test.jsonl", encoding="utf-8").read().splitlines()
item = json.loads(lines[ITEM])
print("reference:", item["raw"])
show(session.interpret(item["path"], realtime=True))
'''),
    ("md", """### Optional: the browser UI through Colab's proxy (experimental)
The full web interface (`serve`) can be opened through Colab's authenticated port proxy. This path has not
been tested on Colab yet; the cells above are the supported demo. It loads a second copy of the models, so run
the last cell first on a T4."""),
    ("code", '''# @title 7. (Experimental) start the web UI
PORT = 8000
url = colab.proxy_url(PORT)
server = subprocess.Popen([sys.executable, "-m", "src.cli", "serve", "--profile", PROFILE, "--port", str(PORT),
                           "--set", f"server.allowed_origins=[{url}]"],
                          stdout=open("/tmp/swi-serve.log", "w"), stderr=subprocess.STDOUT)
print("starting... open", url, "once /tmp/swi-serve.log says the server is ready")
'''),
    ("code", '''# @title 8. Stop: unload the models, free the GPU, optionally delete recordings
DELETE_RECORDINGS = False  # @param {type:"boolean"}
if "server" in globals():
    server.terminate()
summary = session.close()
print("session counters:", summary and summary.get("counters"))
del session
print("GPU memory after cleanup:", colab.free_gpu_memory())
if DELETE_RECORDINGS:
    import shutil
    shutil.rmtree(os.path.join(info["root"], "inputs"), ignore_errors=True)
    print("recordings deleted")
'''),
]

# --------------------------------------------------------------------------------------------------
NOTEBOOKS["06_complete_pipeline.ipynb"] = [
    header("06_complete_pipeline.ipynb", "06 · Complete pipeline: batch, real-time latency and full benchmark", """
Runs the same evaluations as the CPU baseline in the README, on the GPU, so the two are directly comparable:

| Evaluation | Data | CPU baseline (development PC) |
|---|---|---|
| Speech → English (ASR + MT) | all 487 FLEURS `sw_ke` test recordings, human English from FLORES-200 | WER 0.272, chrF 48.4 |
| Real-time latency, TTS on | 30 FLEURS recordings (from #401) streamed at speaking pace | first audio 3.44 / 5.12 / 5.70 s (p50/p95/p99) |
| Text translation | FLORES-200 devtest, 1012 sentences | BLEU 39.0, chrF 61.6 |
| Robustness (optional) | 60 recordings × 9 audio conditions | clean 0.257 … 5 dB noise 0.732 |

Then a batch interpretation of your own file with every artifact exported (transcript, translation, English
speech, events, manifest) and one ZIP of everything. Full run on a T4: roughly 30-60 minutes.
"""),
    ("code", setup_cell(', "--no-smoke"')),
    ("code", '''# @title 2. What to run
SPEECH_N = 487  # @param {type:"integer"}
LATENCY_N = 30  # @param {type:"integer"}
MT_N = 1012  # @param {type:"integer"}
ROBUSTNESS = False  # @param {type:"boolean"}
ROBUSTNESS_N = 60  # @param {type:"integer"}
'''),
    ("code", '''# @title 3. Speech → English (ASR WER + speech-to-English BLEU/chrF + oracle MT)
runs = []
r = colab.run_cli(["evaluate", "speech", "--n", str(SPEECH_N), "--profile", PROFILE], prefix="eval-speech")
runs.append({"label": "speech", **r})
'''),
    ("code", '''# @title 4. Real-time latency (streamed at speaking pace, TTS on)
r = colab.run_cli(["evaluate", "latency", "--n", str(LATENCY_N), "--offset", "400", "--profile", PROFILE],
                  prefix="eval-latency")
runs.append({"label": "latency", **r})
'''),
    ("code", '''# @title 5. Text translation (FLORES-200 devtest)
r = colab.run_cli(["evaluate", "mt", "--n", str(MT_N), "--profile", PROFILE], prefix="eval-mt")
runs.append({"label": "mt", **r})
'''),
    ("code", '''# @title 6. (Optional) robustness: 9 audio conditions
if ROBUSTNESS:
    r = colab.run_cli(["evaluate", "speech", "--n", str(ROBUSTNESS_N), "--no-oracle", "--bootstrap", "500",
                       "--profile", PROFILE, "--conditions", "clean", "noise_20db", "noise_10db", "noise_5db",
                       "quiet_-20db", "reverb_0.6s", "fast_1.15x", "slow_0.85x", "telephone"], prefix="eval-speech")
    runs.append({"label": "robustness", **r})
'''),
    ("code", '''# @title 7. Summary next to the CPU baseline
''' + TABLE_HELPERS + '''
df = table(runs)
display(df.dropna(axis=1, how="all"))
csv = colab.results_dir() / f"colab_complete_{GPU}.csv"
df.to_csv(csv, index=False)
cpu = pd.DataFrame([
    {"measure": "Swahili WER (487)", "cpu_baseline": 0.272},
    {"measure": "speech→English chrF (487)", "cpu_baseline": 48.38},
    {"measure": "first audio p50 s", "cpu_baseline": 3.44}, {"measure": "first audio p95 s", "cpu_baseline": 5.12},
    {"measure": "first audio p99 s", "cpu_baseline": 5.70},
    {"measure": "FLORES BLEU", "cpu_baseline": 39.03}, {"measure": "FLORES chrF", "cpu_baseline": 61.59}])
def pick(label, col):
    sub = df[df.label == label]
    if "condition" in sub:
        sub = sub[sub.condition.isna() | (sub.condition == "clean")]
    return sub[col].iloc[0] if col in sub and len(sub) else None
cpu["gpu"] = [pick("speech", "wer"), pick("speech", "s2t_chrf"), pick("latency", "first_audio_p50_s"),
              pick("latency", "first_audio_p95_s"), pick("latency", "first_audio_p99_s"), pick("mt", "bleu"), pick("mt", "chrf")]
display(cpu)
print("Note: the GPU run uses the colab profile (MT beam 4, ASR partials on); the CPU baseline used MT beam 2.")
'''),
    ("code", '''# @title 8. Batch: interpret your own file with every artifact exported
UPLOAD = False  # @param {type:"boolean"}
INPUT = "data/fleurs_sw/test/" + sorted(os.listdir("data/fleurs_sw/test"))[0]  # default: a FLEURS recording
if UPLOAD:
    INPUT = str(colab.upload_audio(os.path.join(info["root"], "inputs"))[0])
r = colab.run_cli(["interpret", "--input", INPUT, "--export", "--quiet", "--profile", PROFILE], prefix="interpret")
runs.append({"label": "interpret", **r})
from IPython.display import Audio, display
if r["run_dir"]:
    from pathlib import Path
    for f in ("transcript_sw.txt", "translation_en.txt"):
        p = Path(r["run_dir"]) / f
        if p.exists():
            print(f, "->", p.read_text(encoding="utf-8").strip()[:500])
    wav = Path(r["run_dir"]) / "speech_en.wav"
    if wav.exists():
        display(Audio(str(wav)))
'''),
    ("code", '''RUN_DIRS, EXTRA_FILES, EXPORT_NAME = [r["run_dir"] for r in runs], [csv], "complete-pipeline"
''' + EXPORT),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (OUT / name).write_text(json.dumps(nb(name, cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
