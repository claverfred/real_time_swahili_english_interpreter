# Real-Time Swahili–English Interpretation System

## 1. Purpose

Build a professional, reproducible, and modular real-time Swahili-to-English interpretation system. The system must convert live or recorded Swahili speech into accurate English text and natural English speech while minimizing end-to-end latency.

The completed system must support:

1. Google Colab GPU execution
2. Local CPU execution on a Windows PC
3. Local GPU execution when a GPU is available
4. Nelson Mandela University or NM-AIST HPC execution through an institutional account
5. Slurm-based training and fine-tuning
6. Batch interpretation
7. Segmented interpretation
8. Live or near-real-time microphone interpretation
9. Secure SSH-tunneled demos and Jupyter sessions
10. Offline open-source operation and cloud/API fallbacks

The development objective is not to create a collection of disconnected scripts. It is to deliver a maintainable repository with shared components, reproducible benchmarks, human-quality evaluation, HPC training workflows, tests, documentation, and deployment interfaces.

---

## 2. Core Pipeline

```text
Swahili audio
  -> capture and resampling
  -> voice activity detection
  -> streaming ASR
  -> stabilized Swahili transcript
  -> context-aware machine translation
  -> English text chunking
  -> streaming text-to-speech
  -> English audio playback or download
```

### Required operating modes

- **Batch mode:** complete audio files are interpreted and exported
- **Segmented streaming mode:** speech is processed in overlapping segments
- **Live mode:** live microphone or WebSocket audio is processed continuously
- **ASR-only mode:** Swahili transcription
- **MT-only mode:** Swahili text to English text
- **TTS-only mode:** English text to English speech
- **Evaluation mode:** accuracy and latency benchmarks
- **Training mode:** HPC fine-tuning or continued training

A batch-only implementation must not be described as real-time.

---

## 3. System Architecture

Use a modular asynchronous architecture:

```text
Audio Capture
  - microphone
  - uploaded file
  - network or WebSocket stream
       |
Audio Preprocessing
  - 16 kHz mono PCM
  - resampling
  - buffering
  - level and format validation
       |
Voice Activity Detection
  - speech start/end detection
  - configurable silence thresholds
       |
Streaming ASR
  - partial hypotheses
  - stable prefixes
  - final segments
       |
Text Stabilization
  - punctuation
  - normalization
  - repeated-text removal
  - uncertainty handling
       |
Translation Queue
  - context management
  - sentence aggregation
       |
Machine Translation
  - partial or final translation
  - batched or streaming inference
       |
Streaming Text-to-Speech
  - sentence-safe chunks
  - first-audio optimization
       |
English Playback
  - browser, API, local speaker, or file
```

Implement producer-consumer queues or equivalent asynchronous tasks.

Record every stage with timestamped event IDs:

```text
audio_frame_received
vad_speech_start
vad_speech_end
asr_partial
asr_final
transcript_stabilized
mt_requested
mt_partial
mt_final
tts_chunk_requested
tts_first_audio
tts_chunk_complete
tts_complete
pipeline_complete
```

---

## 4. Model and Tool Research Requirements

Research the official documentation and primary repositories before selecting models.

### ASR candidates

Evaluate, with Swahili support explicitly verified:

- Whisper large-v3
- Whisper large-v3-turbo
- Distil-Whisper variants
- faster-whisper CTranslate2 builds
- NVIDIA Parakeet
- NVIDIA Canary or NeMo speech models
- Meta MMS ASR
- SeamlessM4T or SeamlessStreaming
- Google USM-derived or accessible Gemini audio variants
- Azure, Google, AWS, or Deepgram commercial APIs as optional baselines

Evaluate FP16, BF16, INT8, CPU, GPU, streaming, and quantized configurations.

ASR requirements:

- partial and final hypotheses
- stable-prefix handling
- configurable chunk and overlap length
- silence-based finalization
- maximum utterance duration
- language forcing
- hallucination suppression
- repeated segment removal
- optional word timestamps
- optional word confidence
- no fabricated text for uncertain speech
- measured cold-start and warm-start latency
- separate model-load time from inference time

### Translation candidates

Evaluate:

- NLLB-200-distilled-600M
- NLLB-200-1.3B or 3.3B where hardware permits
- Helsinki-NLP MarianMT Swahili-English models
- MADLAD-400
- M2M-100
- mBART
- SeamlessM4T
- licensed multilingual instruction or reranking models
- LoRA or domain-adapted variants
- commercial translation APIs as optional baselines

Translation requirements:

- forced target-language behavior
- batched and streaming translation
- context-window management
- sentence and clause aggregation
- stable-prefix-only translation
- configurable beam width
- named entity preservation
- number preservation
- acronym preservation
- glossary support
- translation cache
- incomplete-text handling
- fallback model selection
- no completion of uncertain partial speech as fact

### TTS candidates

Evaluate:

- Coqui XTTS-v2
- Edge-TTS
- Azure Neural TTS
- Amazon Polly
- Google Cloud TTS
- Facebook MMS TTS
- Piper
- Kokoro and other streaming open-source TTS implementations
- Orpheus or newer streaming TTS systems
- ElevenLabs or equivalent APIs as optional baselines

TTS requirements:

- sentence-safe chunk boundaries
- streaming synthesis
- first-audio-latency measurement
- chunk overlap or smoothing
- English voice consistency
- number, acronym, and name handling
- SSML or equivalent controls where available
- pronunciation dictionary support
- WAV and PCM output
- loudness and silence validation
- consent and licensing for every voice
- no voice cloning without explicit permission

### VAD candidates

Evaluate:

- Silero VAD
- WebRTC VAD
- Pyannote VAD if licensed appropriately
- faster-whisper built-in VAD
- NeMo VAD
- model-integrated VAD

Expose thresholds for speech detection, minimum speech duration, minimum silence duration, padding, hangover, maximum utterance duration, and turn handling.

---

## 5. Local CPU-Only PC

The user's local Windows PC has no GPU and must remain useful.

Implement:

- CPU-only profiles
- INT8 ASR inference
- lightweight ASR candidates
- Helsinki or quantized MT inference
- Edge-TTS or Piper fallback
- direct Windows microphone capture
- WAV file interpretation
- local FastAPI or browser UI
- optional Gradio or desktop UI
- offline operation after setup
- automatic CPU thread selection
- RAM and disk diagnostics
- graceful out-of-memory handling
- clear CPU-versus-GPU latency reporting

Recommended setup:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[local,asr,mt,tts,audio,ui,dev]"
python -m src.cli diagnose
python -m src.cli interpret --input audio.wav --profile cpu
```

The local UI must support microphone selection, input audio upload, output device selection, live Swahili text, live English text, stage timing, model status, export, and cancellation.

Diagnose missing CUDA, FFmpeg, sound device libraries, PyTorch compatibility, audio drivers, and Python packages.

---

## 6. Google Colab Mode

The system must clone or update from GitHub and use the committed modules rather than notebook duplication.

Colab requirements:

- fresh-runtime setup
- GPU detection
- T4, L4, A100, and L40S support
- CUDA, PyTorch, package, and package-lock reporting
- nvidia-smi diagnostics
- Hugging Face login through Colab Secrets
- Google Drive cache persistence
- browser microphone support where practical
- Hugging Face cache reuse
- model download fallback
- runtime restart handling
- uploaded and Google Drive audio
- output export to ZIP
- JSON, CSV, audio, log, image, and provenance export
- batch and live demos
- GPU-memory cleanup
- dependency smoke test
- no hard-coded session paths
- no credentials in notebooks or source

Provide dedicated notebooks:

- environment setup
- ASR benchmark
- MT benchmark
- TTS benchmark
- live demo
- complete pipeline
- ablation and optimization study

---

## 7. Nelson Mandela University / NM-AIST HPC Mode

The system must support request and use of the Nelson Mandela University/NM-AIST HPC cluster when an account and resource allocation are provided.

The implementation must not invent cluster hosts, partitions, module names, account names, or storage paths. Cluster information must be supplied or validated by an HPC administrator.

Required support:

- SSH access
- VS Code Remote SSH
- JupyterLab
- SSH port forwarding
- Slurm or the institution's native scheduler
- Conda, Mamba, uv, or existing environment systems
- Singularity or Apptainer where permitted
- login-node and compute-node setup
- GPU partition allocation
- CPU partition allocation
- scratch, project, home, and archive storage
- Hugging Face and model cache storage
- private dataset access
- multi-GPU training
- distributed training
- checkpoint recovery
- Slurm job arrays
- experiment tracking
- secure SSH tunnels
- result synchronization back to local PC
- resource utilization reporting
- queue and wall-time reporting
- preemption recovery
- storage and quota reporting
- policy-compliant data handling

Before deployment, verify:

- login host
- SSH username
- authentication method
- project or account code
- GPU and CPU partitions
- GPU types and memory
- CUDA modules
- Python modules
- Conda location
- scratch location
- project storage location
- storage quotas
- wall-time limits
- Internet policy on login and compute nodes
- Hugging Face and container policy
- data-transfer method
- private-data restrictions
- maximum GPU count
- approved containers
- Jupyter or tunnel policy

### Example generic Slurm structure

```bash
#!/bin/bash
#SBATCH --job-name=swahili-interpretation
#SBATCH --account=YOUR_PROJECT_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load YOUR_CUDA_MODULE
module load YOUR_PYTHON_MODULE

source /path/to/conda/etc/profile.d/conda.sh
conda activate swahili_interpreter

export HF_HOME=${SCRATCH_DIR:-$HOME}/hf-cache
export TORCH_HOME=${SCRATCH_DIR:-$HOME}/torch-cache

nvidia-smi
python -m src.cli diagnose
python -m src.cli benchmark --profile hpc
```

Create HPC scripts for:

- benchmark submission
- ASR fine-tuning
- MT fine-tuning
- full evaluation
- Slurm array evaluation
- interactive GPU allocation
- Jupyter startup
- result export
- cache preparation
- dataset validation
- result synchronization
- job monitoring and cancellation

Example local synchronization commands must use placeholders:

```powershell
ssh USERNAME@CLUSTER_LOGIN_HOST
scp -r ./results USERNAME@CLUSTER_LOGIN_HOST:/scratch/USERNAME/project/
scp USERNAME@CLUSTER_LOGIN_HOST:/scratch/USERNAME/project/results.zip .
```

Exact cluster commands must be approved by the institution.

### HPC fine-tuning

Fine-tuning must be triggered only when the base model benchmark, data audit, and validation design justify it.

Support:

- ASR fine-tuning
- MT fine-tuning
- LoRA or QLoRA
- full fine-tuning where justified
- multi-GPU distributed training
- gradient accumulation
- gradient checkpointing
- mixed precision
- checkpointing
- early stopping
- resume after preemption
- validation after every checkpoint
- independent test-set preservation
- speaker-disjoint and session-disjoint splits
- domain-disjoint splits
- test-set leakage detection
- reproducible random seeds
- Hyperparameter sweeps
- Ray Tune or Optuna if available
- Hugging Face Accelerate or Trainer
- DeepSpeed or FSDP if supported
- model and data checksums
- licensing manifests
- no private data in public Git repositories

ASR fine-tuning candidates include Whisper, Distil-Whisper, Parakeet, NeMo, and compatible MMS variants.

MT fine-tuning candidates include NLLB-200-distilled-600M, NLLB-1.3B, Helsinki-Marian, MADLAD-400, mBART, and licensed multilingual variants.

Fine-tuning configuration must include exact base revision, data manifests, learning rate, optimizer, scheduler, warm-up, effective batch size, epochs, LoRA rank and target modules, precision, gradient accumulation, sequence length, validation schedule, checkpoint interval, random seed, and distributed strategy.

Do not fine-tune based only on training-loss reduction. Select a checkpoint using independent validation and human-reference test metrics.

### HPC data policy

- separate public, licensed, restricted, and private datasets
- use checksums and immutable snapshots
- document consent and licensing
- restrict private audio access
- anonymize or encrypt sensitive recordings
- use only approved external APIs
- never send restricted audio to public cloud ASR or TTS without authorization
- preserve private data outside Git
- define deletion and retention policies
- prevent credentials from appearing in logs
- include data manifests with every run

### HPC secure tunnels

Support Jupyter, TensorBoard, MLflow, Gradio, FastAPI, and WebSocket demonstrations through SSH tunnels.

Examples:

```powershell
ssh -N -L 8888:localhost:8888 USERNAME@CLUSTER_LOGIN_HOST
ssh -N -L 7860:localhost:7860 USERNAME@CLUSTER_LOGIN_HOST
```

Do not expose an unauthenticated public server. Tunnel settings must be confirmed under institutional policy.

---

## 8. Shared Runtime and Platform Abstraction

Create shared implementations for:

- configuration
- audio capture and playback
- GPU and CPU diagnosis
- VAD
- ASR
- text normalization
- translation
- TTS
- pipeline orchestration
- event handling
- logging
- storage
- caching
- secrets
- result export
- evaluation
- benchmark execution
- fallback policy
- API and CLI interfaces

Keep platform-specific code thin.

Suggested structure:

```text
src/platform/colab.py
src/platform/local.py
src/platform/hpc.py
src/platform/cuda.py
src/platform/cpu.py
src/platform/audio_capture.py
src/platform/audio_playback.py
src/platform/storage.py
src/platform/secrets.py
```

Colab, local, and HPC modes must share the same core pipeline and model interfaces.

Platform selection must support:

```text
explicit CLI or configuration override
  -> environment override
  -> platform configuration
  -> CUDA auto-detection
  -> CPU fallback
```

No fallback may occur silently. Record every fallback in the manifest and output.

---

## 9. Repository Structure

Create a professional repository:

```text
real-time-swahili-english/
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements.txt
├── requirements-colab.txt
├── requirements-hpc.txt
├── environment.yml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── Makefile
├── configs/
│   ├── default.yaml
│   ├── accuracy.yaml
│   ├── balanced.yaml
│   ├── realtime.yaml
│   ├── low_vram.yaml
│   ├── cpu.yaml
│   ├── offline.yaml
│   ├── colab.yaml
│   ├── local_windows.yaml
│   ├── hpc.yaml
│   └── hpc_finetune.yaml
├── src/
│   ├── capture/
│   ├── audio/
│   ├── vad/
│   ├── asr/
│   ├── text/
│   ├── mt/
│   ├── tts/
│   ├── streaming/
│   ├── pipeline/
│   ├── eval/
│   ├── api/
│   ├── cli/
│   ├── platform/
│   └── utils/
├── notebooks/
│   ├── 01_environment_setup.ipynb
│   ├── 02_model_benchmark.ipynb
│   ├── 03_asr_evaluation.ipynb
│   ├── 04_mt_evaluation.ipynb
│   ├── 05_tts_evaluation.ipynb
│   ├── 06_realtime_demo.ipynb
│   └── 07_hpc_finetuning.ipynb
├── scripts/
│   ├── benchmark.py
│   ├── evaluate.py
│   ├── setup_colab.py
│   └── hpc/
├── tests/
├── data/
├── results/
├── docs/
└── .github/workflows/
```

Avoid machine-specific absolute paths.

---

## 10. Configuration Profiles

Provide the following profiles:

- `accuracy`: prioritize quality
- `balanced`: practical default
- `realtime`: minimize first-audio delay
- `low_vram`: constrain GPU memory
- `cpu`: local CPU-only operation
- `offline`: no external API calls
- `colab`: hosted GPU behavior
- `local_windows`: Windows audio and CUDA behavior
- `hpc`: cluster allocation and cache behavior
- `hpc_finetune`: training behavior
- `hpc_evaluate`: distributed or batch evaluation
- `cloud_fallback`: permitted cloud fallback behavior

Every resolved configuration must include:

- environment
- device
- precision
- ASR model and revision
- MT model and revision
- TTS model or provider and voice
- VAD thresholds
- chunk sizes
- overlap
- beam search size
- translation context window
- maximum tokens
- streaming buffer size
- queue limits
- retry count
- timeout
- cache location
- fallback policy
- license status
- output paths
- logging policy
- privacy policy
- data-retention setting

Support environment and CLI overrides.

---

## 11. Public API and CLI

Provide a Python streaming API similar to:

```python
interpreter = RealtimeSwahiliInterpreter(config)
await interpreter.start()

await interpreter.push_audio(audio_chunk, sample_rate=16000)

async for event in interpreter.events():
    if event.type == "asr_partial":
        print("Partial Swahili:", event.text)

    if event.type == "asr_final":
        print("Swahili:", event.text)

    if event.type == "mt_final":
        print("English:", event.text)

    if event.type == "tts_audio":
        play_audio(event.audio)
```

Support session start, audio push, flush, cancel, reset, status, health, model status, and shutdown.

Provide CLI operations such as:

```powershell
python -m src.cli diagnose
python -m src.cli benchmark --profile realtime
python -m src.cli transcribe --input audio.wav --profile balanced
python -m src.cli interpret --input audio.wav --profile realtime
python -m src.cli evaluate --dataset flores200
python -m src.cli serve --host 127.0.0.1 --port 8000
```

For HPC, use dedicated Slurm entry points and scripts rather than interactive CLI processes on login nodes.

---

## 12. ASR Evaluation

Use licensed, documented Swahili datasets such as:

- Mozilla Common Voice Swahili
- FLEURS Swahili
- public parliamentary or news recordings with suitable rights
- consented custom recordings
- consented domain-specific recordings
- institutional or academic corpora permitted for research

Report:

- WER
- CER
- normalized WER if normalization is disclosed
- number and name recall
- language confidence
- empty result rate
- hallucination and repetition rate
- RTF
- mean, median, p95, and p99 latency
- VRAM and RAM use
- cold and warm start
- segment timing error
- code-switch accuracy
- failure examples
- data provenance
- dataset revision
- sample count and duration
- test split

Do not report only an average.

---

## 13. Translation Evaluation

Use independent human references.

Preferred dataset:

- FLORES-200 Swahili-English devtest

Use separate sources and splits for training and testing.

Report:

- BLEU
- chrF
- chrF++
- COMET if compatible
- BERTScore
- BLEURT if feasible
- human adequacy
- human fluency
- terminology quality
- named entity preservation
- number preservation
- acronym preservation
- omission rate
- hallucination rate
- translation latency
- p95 and p99 latency
- VRAM and RAM use
- partial-translation stability
- context-window benefit
- fine-tuning benefit

Never generate a hypothesis and use it as the reference. Never compare one MT model against another model's output and call that ground truth.

A 100% BLEU score is invalid as a human benchmark if hypotheses are also references. Use explicit human-translated references and disclose every metric method.

Human evaluation should assess adequacy, fluency, named entities, numbers, terminology, mistranslations, and severity.

---

## 14. TTS and End-to-End Evaluation

Report:

- RTF
- first-audio latency
- complete synthesis latency
- p95 and p99 latency
- audio duration
- output format
- sample rate
- VRAM and RAM use
- listening study score if feasible
- intelligibility
- naturalness
- prosody
- pronunciation of names and numbers
- chunk boundary artifacts
- network reliability for cloud TTS
- fallback behavior

End-to-end evaluation must measure:

- capture time
- VAD time
- ASR partial time
- ASR final time
- MT start and completion
- TTS start
- first English audio
- full English audio
- queue wait
- playback delay
- GPU memory
- CPU memory
- network latency
- jitter
- underruns
- overflows
- ASR revisions
- MT revisions
- total pipeline time

Test:

- short sentences
- long paragraphs
- questions
- fast and slow speech
- quiet speech
- noisy speech
- multiple speakers
- code-switching
- names and numbers
- technical terms
- interruptions
- silence
- poor microphones
- reverberation
- incomplete sentences
- very long utterances
- network failure
- GPU failure
- runtime interruption
- out-of-memory conditions
- private and restricted audio

---

## 15. Optimization and Profiles

Profile before optimizing.

Measure and optimize:

- model load
- VAD overhead
- ASR chunk size
- beam size
- precision
- VAD padding
- translation batch size
- context-window size
- TTS chunk size
- audio concatenation
- queue wait
- GPU synchronization
- memory copies
- thread settings
- CUDA kernels
- network round trips
- cache reuse
- checkpoint recovery
- tail latency

Required profiles:

| Profile | Priority | GPU | Intended use |
|---|---|---:|---|
| accuracy | highest quality | required for larger models | offline research |
| balanced | quality and latency | GPU recommended | default research/demo |
| realtime | first-audio latency | GPU recommended | live interpretation |
| low_vram | memory efficiency | 8–16 GB or quantized | Colab or local GPU |
| cpu | operation without GPU | none | local Windows PC |
| hpc | benchmark and throughput | data-center GPU | HPC cluster |
| hpc_finetune | training | allocated GPU | HPC fine-tuning |
| offline | no external calls | optional | private or restricted audio |

Do not report throughput without first-audio latency and accuracy.

---

## 16. Security and Privacy

Implement:

- environment-variable and platform-secret authentication
- no credentials in source or notebooks
- no credentials in Slurm scripts or logs
- input size limits
- temporary audio cleanup
- configurable retention
- session expiration
- rate limiting for network APIs
- WebSocket authentication where deployed
- HTTPS or institution-approved access for remote servers
- no unauthenticated public GPU server
- clear consent for microphone use
- cloud-data warnings
- restricted-data warnings
- license checks
- download-integrity validation
- audit logs without private audio unless requested
- safe shutdown
- secure SSH-tunnel setup
- no private datasets in public Git repositories

---

## 17. Testing

Unit tests must cover:

- audio loading and resampling
- VAD state transitions
- ASR serialization
- stabilized text behavior
- MT batching
- context retention
- TTS chunking
- audio concatenation
- queue behavior
- cancellation and reset
- timeout and retry
- fallback selection
- configuration validation
- empty transcription
- malformed input
- Unicode and Swahili punctuation
- names, numbers, and acronyms
- partial and final events
- export consistency
- result manifest consistency

Integration tests must cover:

- short Swahili file
- noisy Swahili file
- code-switched file
- simulated live stream
- ASR benchmark
- MT evaluation
- TTS synthesis
- complete pipeline
- runtime interruption
- model fallback
- Colab setup
- local CPU setup
- HPC smoke test
- result synchronization
- SSH tunnel setup where policy permits
- clean export and cleanup

Use CI for linting and lightweight tests. Do not run expensive model benchmarks in ordinary CI.

---

## 18. Documentation

Create:

- README
- architecture guide
- model-selection guide
- dataset methodology
- evaluation methodology
- latency methodology
- streaming protocol
- Colab guide
- local Windows guide
- HPC request and setup guide
- Slurm guide
- SSH tunnel guide
- fine-tuning guide
- API reference
- CLI reference
- configuration reference
- troubleshooting guide
- privacy and consent guide
- licensing guide
- limitations
- ethics and bias analysis
- reproducibility guide
- thesis findings
- system card or model card

Document cluster-specific details only after institutional verification.

---

## 19. Required Master Prompt

The coding agent must implement the following request:

```text
Act as a senior speech-processing engineer, machine-learning engineer, distributed-computing engineer, and production software architect.

Build a complete real-time Swahili-to-English interpretation system for:

- Google Colab
- a CPU-only Windows PC
- a GPU PC
- Nelson Mandela University/NM-AIST HPC resources

Design and implement a modular cascade pipeline:

Live or uploaded Swahili audio
  -> preprocessing
  -> voice activity detection
  -> streaming ASR
  -> Swahili stabilization
  -> context-aware Swahili-English MT
  -> English text chunking
  -> streaming English TTS
  -> English playback or download

Research and verify current models and tools before implementation. Evaluate at least:

- faster-whisper or a stronger Swahili ASR model
- CTranslate2 or TensorRT variants if stable
- NVIDIA NeMo or Parakeet candidates where supported
- Whisper, Whisper-turbo, or distilled variants
- NLLB-200
- Helsinki-NLP MarianMT
- MADLAD or M2M where suitable
- Seamless speech translation candidates
- Coqui XTTS-v2
- Edge-TTS
- open-source streaming TTS alternatives
- Silero VAD

Do not claim Swahili support without testing or primary-source evidence.

Create shared source code for Colab, local PC, and HPC environments. Platform-specific code must remain thin wrappers. The local profile must operate on CPU. The Colab profile must operate on hosted GPU. The HPC profile must submit and resume jobs through Slurm or the institution's scheduler.

For the local PC:

- support Windows and PowerShell
- support CPU inference
- support Python virtual environments
- support direct microphone capture
- support Gradio, FastAPI, or desktop UI
- diagnose missing audio and runtime dependencies
- provide a low-latency profile and a quality profile
- avoid hard-coded paths

For Google Colab:

- support a fresh cloned repository
- detect CUDA and VRAM
- install verified dependencies
- support Hugging Face secrets
- support Google Drive caches
- support live, uploaded, and file-streamed audio
- export complete result artifacts
- provide clean notebook setup and smoke tests

For the Nelson Mandela University/NM-AIST HPC:

- require institutional account and resource approval
- verify cluster hostname, partitions, modules, storage, quotas, and data policy
- provide generic placeholders until approved details are supplied
- support SSH and VS Code Remote SSH
- support Jupyter through SSH tunnels
- provide Slurm scripts and dry-run validation
- support ASR and MT fine-tuning with LoRA or QLoRA
- support distributed training where permitted
- support checkpointing and preemption recovery
- synchronize datasets, caches, checkpoints, logs, and results
- prevent private datasets and credentials from entering Git
- comply with institutional licensing and restricted-data policy
- report queue, node, GPU, CPU, memory, wall-time, and scratch-I/O metrics

Implement:

- robust VAD
- streaming ASR with partial and final text
- Swahili text stabilization
- context-aware MT
- streaming TTS
- low-latency queueing
- model fallback
- event tracing
- JSON and CSV outputs
- structured logs
- progress and health status
- cancellation and reset
- resource monitoring
- export and download
- local API or WebSocket support
- batch and live entry points
- unit and integration tests
- CI for linting and lightweight tests

Create independent ASR, MT, TTS, and end-to-end benchmarks.

Use human-translated FLORES-200 or an equivalent independent reference for MT evaluation. Do not use model-generated hypotheses as references. Never report 100.0 BLEU merely because a model output was tested against itself.

Measure:

- WER and CER
- BLEU and chrF
- human adequacy and fluency where feasible
- COMET or BERTScore when feasible
- TTS RTF and first-audio latency
- end-to-end first-audio latency
- mean, median, p95, and p99 latency
- VRAM and RAM use
- cold and warm start
- queue wait
- interrupted-stream behavior
- ASR and MT revision behavior
- fallback behavior

Compare and select models using accuracy, latency, memory, license, supported hardware, and operational reliability. Do not assume that the largest model is the best system.

Profile at least:

- accuracy
- balanced
- realtime
- low-vram
- cpu
- offline
- colab
- local-windows
- hpc
- hpc-finetune

Document every model revision, package version, benchmark setting, hardware target, data source, license, test split, and random seed.

Do not fabricate results. Do not hide failed candidates. Do not overwrite baseline measurements. Do not claim real-time behavior from batch-only inference.

Before changing a model, quantify the result:

- accuracy gain
- latency cost
- memory cost
- stability cost
- licensing impact
- hardware compatibility
- operational fallback cost

The final output must include:

1. repository summary
2. architecture diagram
3. model-selection report
4. candidate comparison tables
5. ASR, MT, TTS, and end-to-end benchmarks
6. local PC setup
7. Colab setup
8. HPC request and setup documentation
9. Slurm job scripts
10. SSH tunnel instructions
11. fine-tuning procedure
12. API and CLI documentation
13. tests and CI
14. human evaluation methodology
15. privacy and licensing report
16. limitations and bias analysis
17. reproducibility instructions
18. final accuracy-first configuration
19. final balanced configuration
20. final real-time configuration
21. final CPU configuration
22. thesis-ready findings
23. recommended future work
24. concise executive summary
```

---

## 20. Deployment Matrix

| Environment | GPU | CPU | Training | Fine-tuning | Primary purpose |
|---|---|---|---|---|---|
| Local Windows PC | Optional | Required | No or limited | No or limited | Demo and batch/streaming CPU operation |
| Google Colab | Hosted GPU | Available | Limited | Limited | Reproducible prototype and benchmark |
| Local GPU PC | Supported | Available | Optional | Optional | Real-time local deployment |
| Nelson Mandela University/NM-AIST HPC | Allocated cluster GPU | Allocated CPU | Supported | Supported | Full benchmark and fine-tuning |

---

## 21. Acceptance Criteria

The project is complete only if:

- every environment passes the documented smoke test
- the local PC operates without a GPU
- Google Colab runs from a fresh GitHub clone
- HPC jobs are approved and reproducible
- batch and streaming inputs work
- live audio works where platform permissions allow
- ASR, MT, and TTS return structured results
- no generated text is used as an MT reference
- independent human translations are used for final MT evaluation
- end-to-end first-audio latency is measured
- tail latency is measured
- GPU and CPU profiles are documented
- Slurm jobs can checkpoint and recover
- private data and credentials are protected
- fallbacks are logged
- model and data licenses are documented
- tests pass
- documentation is complete
- benchmark artifacts are exportable
- a fresh environment can reproduce the setup
- final recommendations clearly distinguish accuracy, speed, memory, and hardware trade-offs
- unknown cluster settings remain placeholders until institutionally verified
