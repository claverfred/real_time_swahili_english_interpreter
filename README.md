# Real-Time Swahili–English Interpreter

Modular cascade pipeline: Swahili audio → VAD → streaming ASR → stabilization → MT → chunking →
streaming TTS → English audio. One shared codebase for a CPU-only Windows PC, GPU PCs, Google Colab,
and HPC (Slurm). Full requirements: [REALTIME_SWAHILI_ENGLISH_SYSTEM.md](REALTIME_SWAHILI_ENGLISH_SYSTEM.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository scaffold, packaging, CI | done |
| 1 | Config profiles, event tracing, logging, manifests, platform diagnosis, CLI | done |
| 2 | Audio I/O + VAD | done |
| 3 | Streaming ASR | done |
| 4 | Text stabilization + MT | done |
| 5 | Streaming TTS | next |
| 6 | Async pipeline + Python API + batch/live CLI | planned |
| 7 | FastAPI/WebSocket server + local UI | planned |
| 8 | Evaluation and benchmarks | planned |
| 9 | Colab notebooks | planned |
| 10 | HPC Slurm scripts + fine-tuning | planned |
| 11 | Documentation, Docker, release | planned |

No real-speech benchmark results exist yet (Phase 8). Model IDs in `configs/` are candidates until measured.

## Quick start (Windows, PowerShell)

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,audio,local]"
python -m src.cli diagnose
python -m src.cli config list
python -m src.cli config show --profile cpu --resolve
pytest -m "not slow and not gpu"
```

## Audio and VAD (Phase 2)

```powershell
python -m src.cli devices                                   # list microphones / speakers
python -m src.cli record --seconds 5 --output my_clip.wav    # local test recording
python -m src.cli inspect --input my_clip.wav                # format, levels, clipping, silence
python -m src.cli vad --input my_clip.wav --profile cpu      # speech segments
python -m src.cli vad --input my_clip.wav --backend webrtc --export --save-segments
```

VAD backends:

| Backend | Notes |
|---|---|
| `silero` | Silero v5.1.2 run through onnxruntime, with no PyTorch needed. Downloaded once to `cache/models/` and SHA-256 verified. MIT license. |
| `webrtc` | Binary decisions; very fast. |
| `energy` | Dependency-free last resort. |

If a backend can't load, the next one in the chain `silero → webrtc → energy` is used, and the fallback is recorded.

The segmenter keeps `speech_pad_ms` of audio before speech onset. A segment opens only after
`min_speech_ms` of continuous speech, closes after `min_silence_ms` of silence, and keeps
`hangover_ms` of audio after the last speech frame. Utterances longer than `max_utterance_s` are split.
Timestamps are counted in samples, so results don't depend on how the audio is chunked.

## Speech recognition (Phase 3)

```powershell
python -m src.cli transcribe --input clip.wav --profile cpu                       # batch: VAD + final decode per utterance
python -m src.cli transcribe --input clip.wav --profile cpu --live-sim            # real-time-paced stream with partials
python -m src.cli transcribe --mic --seconds 30 --profile cpu                      # live microphone
python -m src.cli transcribe --input clip.wav --model large-v3-turbo --reference "Habari za asubuhi" --export
```

**Engine:** faster-whisper (CTranslate2), with the language forced to `sw`. Download time, model
load time, the first (cold) decode and later (warm) decodes are timed separately. The resolved
Hugging Face snapshot commit is recorded in every manifest.

**Streaming:** partial results come from greedy re-decodes about every `asr.chunk_s` seconds of
new audio. A word prefix is committed once `asr.stable_prefix_agreement` consecutive hypotheses
agree on it. Committed text never shrinks; if a later hypothesis disagrees, that is counted as a
revision. The final result is a full beam-search decode run when VAD closes the utterance. On slow
hardware, partials automatically become less frequent instead of building a backlog.

**Hallucination policy:** the filter removes or flags text and never fills gaps. Each removed
segment records why it was removed:

| Reason | Trigger |
|---|---|
| `empty` | The segment has no text |
| `no_speech` | Whisper's no-speech probability is high and confidence is low |
| `repetition_loop` | Compression ratio is too high (the model is repeating itself) |
| `hallucination_phrase` | A known subtitle or video-outro phrase, e.g. "Asante kwa kutazama" |
| `duplicate_segment` | The text repeats the previous utterance |
| `very_low_confidence` | Average log-probability is far below the threshold |

Low-confidence text that is kept is flagged `uncertain`. Runaway word loops are collapsed.

**Smoke test set:** `python scripts/make_synthetic_swahili.py` creates Edge-TTS Swahili clips with
known reference text. `python scripts/asr_smoke_compare.py --models base small ...` compares
models on them. The clips are synthetic speech, so the results are integration evidence, not
Swahili accuracy figures. Phase 8 benchmarks on FLEURS and Common Voice.

Smoke result on the local CPU (10 cores, int8, 10 clips, 2 voices):

| Model | WER | CER | p50 decode | RTF | Load | RSS |
|---|---|---|---|---|---|---|
| base | 0.936 | 0.187 | 719 ms | 0.14 | 0.5 s | 195 MB |
| small | 0.723 | 0.132 | 2395 ms | 0.49 | 1.5 s | 351 MB |
| medium | 0.415 | 0.076 | 6120 ms | 1.24 | 3.8 s | 900 MB |
| large-v3-turbo | 0.117 | 0.026 | 8555 ms | 1.74 | 3.8 s | 990 MB |

The `cpu` profile uses `small`, the most accurate model that keeps up in real time on this
CPU. Streaming on a 30 s clip gave a median final lag (end of speech to final text) of 2.9 s.
Most of small's errors are word boundaries (for example "Leoninaenda" for "Leo ninaenda"), which
WER penalises heavily; CER is the fairer view. `large-v3-turbo` is far more accurate but slower
than real time on CPU. Use it for batch files, or on a GPU.

## Translation (Phase 4)

```powershell
python scripts/fetch_flores200.py                                      # FLORES-200 dev/devtest (human references)
python -m src.cli translate --text "Habari za asubuhi." --profile cpu  # MT-only mode
python -m src.cli translate --input sentences.txt --glossary terms.csv --mask --export
python scripts/mt_eval.py --profile cpu --models facebook/nllb-200-distilled-600M Helsinki-NLP/opus-mt-swc-en --n 200 --context
```

**Path from ASR to English:** each ASR final goes through stabilization, then sentence
aggregation, then MT.

- **Stabilization** changes only spacing, punctuation, case and runaway repeats. Every change
  is listed, and words are never added or guessed.
- **Sentence aggregation** is Swahili-aware: abbreviations such as `Dkt.`, `Bw.` and `n.k.`
  don't end a sentence. After a max-duration VAD split, the unfinished tail is held until the
  sentence completes. At a real pause, everything is released. Run-on speech is broken at a
  clause boundary after `text.max_sentence_words`.
- **Events:** `transcript_stabilized`, `mt_requested`, `mt_partial`, `mt_final`.

**Engines:**

- **CTranslate2** (default): int8 on CPU. The Hugging Face model is converted once into
  `cache/ct2/`, with a `provenance.json` recording the source commit, quantization and
  ctranslate2 version. Conversion needs transformers and torch; running the converted model
  doesn't.
- **transformers:** the reference engine.

If a model or engine fails, the next one is tried, and the fallback is recorded.

**Safeguards:**

- An LRU cache.
- Partial translation is opt-in (`mt.translate_partials`). Only the *committed* ASR prefix is
  translated, greedily, and the result is marked provisional. The unstable tail is never sent.
- Uncertain transcripts are translated and flagged, or skipped (`mt.uncertain_policy`).
- Output guards flag `length_ratio` (possible hallucinated completion), `repeat_collapsed` and
  `empty_output`.
- Preservation checks raise `number_missing`, `acronym_missing` and `glossary_missing`. Name
  preservation is reported but not flagged: Swahili capitalizes common nouns that are
  correctly translated ("Kamati" becomes "Committee"), so that flag fired on 44% of sentences.
- With `--mask`, numbers and glossary terms are replaced by placeholders (`Z1Q`, the format
  that survived all probe sentences on both models). If the model drops a placeholder, the
  sentence is re-translated unmasked.

**Measured on the local CPU** (FLORES-200 devtest, first 200 sentences, human references,
CTranslate2 int8, beam 2, batch 1; sacrebleu signatures in the run's `results.json`):

| Model | License | BLEU | chrF | chrF++ | p50 | p95 | Numbers kept | RAM |
|---|---|---|---|---|---|---|---|---|
| nllb-200-distilled-600M | CC-BY-NC-4.0 | 42.0 | 64.2 | 62.9 | 719 ms | 1201 ms | 91% | ~845 MB |
| nllb-600M + 1 sentence of context | | 40.8 | 62.9 | 61.5 | 1720 ms | 2900 ms | 87% | |
| opus-mt-swc-en | Apache-2.0 | 21.8 | 46.5 | 45.1 | 116 ms | 201 ms | 81% | ~130 MB |

- The `cpu` profile uses NLLB-600M, with Marian as its fallback. `opus-mt-swc-en` was trained
  on *Congo* Swahili, and it shows: "Dodoma" became "Dodama", and "shilingi" became "pounds".
- Context prefixing lowered scores and doubled latency, so `mt.context_mode` defaults to `none`.
- **NLLB is licensed for non-commercial use only.**

**Cascade caveat:** on clean Swahili text NLLB is good. On Whisper `small` output from this
CPU, word-boundary ASR errors carry through. For example, "elfumbili" ("elfu mbili", two
thousand) became "two shillings". On CPU-only hardware, ASR quality is the end-to-end
bottleneck. GPU profiles (`large-v3-turbo`, ASR WER 0.12 on the smoke set) avoid most of it,
and Phase 8 measures it properly.
## Configuration

Profiles live in `configs/` and inherit through `inherits:`. Precedence (highest first):

1. CLI: `--set asr.beam_size=3` (repeatable)
2. Environment: `SWI__ASR__BEAM_SIZE=3`, plus `SWI_PROFILE`, `SWI_DEVICE`, `SWI_PLATFORM`
3. Profile YAML chain
4. Auto-detection (`device: auto` → CUDA if available, else CPU)

Fallbacks such as CUDA→CPU or bf16→fp16 are never silent. Each one is recorded and goes into the
run manifest. Set `fallback.policy=fail` to make a fallback an error instead.

The config validator refuses:
- cloud backends when `privacy.data_class` is `restricted` or `private`
- cloud backends unless `fallback.allow_cloud` is set
- voice cloning
