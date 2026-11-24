# Running on Google Colab

Six notebooks in [`notebooks/`](../notebooks) run the interpreter on a Colab GPU. They are thin:
every cell calls the committed code (`src/`, `scripts/`, `python -m src.cli …`), so Colab, the
local PC and (later) the HPC cluster run the same pipeline.

> **Status: built, not yet run on Colab.** The notebooks, setup script and helpers were developed
> and tested on the Windows CPU machine: unit tests, a local run of the setup script with its
> smoke test, and the notebook session wrapper on a real recording. No GPU numbers exist yet.
> Notebooks 02, 03 and 06 are there to produce them.

| Notebook | What it does | Time on a T4 (estimate) |
|---|---|---|
| [01_environment_setup](../notebooks/01_environment_setup.ipynb) | Fresh-runtime setup, GPU/CUDA/package report, lock file, smoke test of every stage, unit tests | 10 min first run, 2 min later |
| [02_asr_benchmark](../notebooks/02_asr_benchmark.ipynb) | Swahili ASR candidates on all 487 FLEURS recordings: WER with CIs, decode p50/p95/p99, RTF, peak GPU memory | 30-60 min |
| [03_mt_benchmark](../notebooks/03_mt_benchmark.ipynb) | NLLB 600M / 1.3B / 3.3B and Marian on FLORES-200 devtest: BLEU/chrF with CIs, latency, GPU memory | 30-60 min |
| [04_tts_benchmark](../notebooks/04_tts_benchmark.ipynb) | English voices: first-audio latency, RTF, Whisper round-trip intelligibility, listening samples | 10 min |
| [05_live_demo](../notebooks/05_live_demo.ipynb) | Browser microphone, upload, Drive file, real-time-paced FLEURS clip; English audio plays in the notebook | interactive |
| [06_complete_pipeline](../notebooks/06_complete_pipeline.ipynb) | The Phase 8 benchmark on GPU next to the CPU baseline, optional robustness, batch export, ZIP | 30-60 min |

## Start

1. Open a notebook from GitHub (the badge at its top) or upload it.
2. **Runtime → Change runtime type → GPU.** T4 is enough; L4, A100 and L40S are faster and use bf16.
3. Optional: add a Hugging Face token as the Colab Secret **`HF_TOKEN`** (key icon on the left).
   Every default model is public; a token only raises download rate limits. Tokens are never
   printed, stored in notebooks, or committed.
4. Run the first cell. It is the same in every notebook and is safe to re-run, also after a
   runtime restart.

## What the setup cell does

```text
git clone / git pull  →  colab.bootstrap()                →  scripts/setup_colab.py
                          · mount Google Drive               · pip install requirements-colab.txt
                          · persistent root + SWI_COLAB_ROOT   with Colab's torch/numpy/scipy pinned
                          · link data/ folders to Drive      · pip freeze → env/pip-freeze-*.txt
                          · HF login from Colab Secrets      · environment report (GPU, CUDA, cuDNN)
                          · CUDA libs for CTranslate2        · fetch FLEURS + FLORES once
                                                             · smoke test (notebook 01)
```

- **No runtime restart.** Our extras are installed with constraints that keep Colab's own
  `torch`, `numpy`, `scipy`, `jax` and `tensorflow`. If a restart is still needed, the cell says
  so and names the package.
- **Persistence.** With Drive mounted, everything lives in `MyDrive/swahili_interpreter/`:
  `cache/` (Hugging Face downloads and converted CTranslate2 models), `data/`, `results/`,
  `logs/`, `env/` (setup reports and lock files) and `inputs/` (your recordings). Without Drive
  it is `/content/swahili_interpreter/`, which is lost when the runtime is reset.
- **CUDA libraries.** CTranslate2 (faster-whisper, NLLB) needs cuDNN 9 and cuBLAS 12. Colab's
  PyTorch ships them as pip packages that CTranslate2 does not search. `bootstrap()` adds them to
  `LD_LIBRARY_PATH` for the CLI subprocesses, and `preload_cuda_libraries()` loads them for
  in-notebook use (notebook 05).
- **Model download fallback.** Each stage walks its configured fallback models and logs any
  fallback in the run manifest. Benchmarks set `fallback.policy=fail` instead, so a model that
  cannot load is reported as failed and never silently replaced by another.

## The `colab` profile

[`configs/colab.yaml`](../configs/colab.yaml) uses the models measured on CPU (Mollel Swahili
Whisper-small, NLLB-600M, Piper), so GPU and CPU numbers are directly comparable. On the GPU it
also turns on ASR partials and MT beam 4. Precision is `auto`: fp16 on T4, bf16 on Ampere and
newer. Larger models are compared in notebooks 02 and 03 through `--set` overrides. The profile
is changed only after those measurements exist.

## Evaluations and GPU memory

The benchmark cells call `colab.run_cli([...])`. It runs `python -m src.cli evaluate …` in a
subprocess, streams its output, samples GPU memory with `nvidia-smi` every 0.5 s, and returns the
new run directory. GPU memory is released when the subprocess exits. Each run writes
`results.json`, `report.md`, `items.jsonl` and a manifest with model revisions, dataset revision,
config and runtime. Comparison tables come from
[`src/eval/summary.py`](../src/eval/summary.py) and are saved as CSV and PNG next to the runs.

## Live demo and privacy

Notebook 05 loads the pipeline once on the GPU with `InterpreterSession` (a synchronous wrapper
that works inside the notebook's event loop). It then interprets:

- **Browser microphone:** the browser asks for permission; a *Stop* button ends the recording.
  This is turn-taking: speak, stop, then hear the English.
- **Uploaded files**, files on Drive, and a FLEURS recording fed at speaking pace (listener latency).

Recordings are stored only in the runtime and `MyDrive/swahili_interpreter/inputs/`. The last
cell can delete them. Every model runs inside the Colab runtime; no audio or text is sent to
any other service.

The full browser UI can be opened through Colab's authenticated port proxy (cell 7, experimental,
untested).

## Export

The last cell of each notebook zips that notebook's run directories with the CSV/PNG tables and
the latest environment reports and lock file, then downloads the ZIP. The ZIP is also kept under
`results/` on Drive.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[warn] GPU: running on CPU` | Runtime → Change runtime type → GPU, then re-run the setup cell |
| `Could not load library libcudnn_ops.so.9` | Re-run the setup cell (it sets the CUDA library path); in notebook 05 run cell 2 again |
| "Restart the runtime" printed by setup | Runtime → Restart session, then re-run the setup cell |
| Drive mount refused | Set `USE_DRIVE = False` (nothing persists after the runtime is reset) |
| Out of GPU memory (3.3B model on T4) | Run the "Free GPU memory" / last cell of the other notebook, or use an L4/A100 |
| Hugging Face rate limit (HTTP 429) | Add the `HF_TOKEN` Colab Secret |

The notebooks are generated by [`scripts/make_notebooks.py`](../scripts/make_notebooks.py). Edit
that file, run it, and commit both. A test checks that the committed notebooks are up to date and
contain no outputs or tokens.
