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
| 5 | Streaming TTS | done |
| 6 | Async pipeline + Python API + batch/live CLI | done |
| 7 | FastAPI/WebSocket server + local UI | in progress |
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
## Speech synthesis (Phase 5)

```powershell
python -m src.cli speak --text "Good morning. The price is 2,500 shillings." --output out.wav --profile cpu
python -m src.cli speak --text "Hello" --backend sapi --play
python -m src.cli speak --text "Hello" --backend edge --voice en-TZ-ImaniNeural --profile local_windows   # cloud, opt-in
python scripts/tts_eval.py --profile cpu --candidates piper:en_US-libritts_r-medium sapi:default
```

**Path from MT to audio:** only *final* translations are spoken, because audio can't be taken
back once played.

- **Chunking:** English text is split only at sentence ends, so "Dr." and "U.S." don't break
  it. Over-long sentences are split at clause boundaries. The first chunk may end at an early
  clause boundary so audio starts sooner.
- **Spoken form:** acronyms are spelled out ("CCM" becomes "C C M") unless they are normally
  read as words. "TSh 2,000" becomes "2,000 Tanzanian shillings", and "%" becomes "percent".
  An optional pronunciation dictionary is applied last.
- **Consistent output:** every backend's audio is resampled to `tts.sample_rate`, trimmed of
  silence, normalized to -20 dBFS RMS with a -1 dBFS peak limit, and faded at chunk edges to
  avoid clicks. Chunks are joined with `tts.sentence_pause_ms` of silence.
- **Validation:** each chunk is checked for empty output, clipping, very quiet audio, mostly
  silent audio, and an implausible duration for the text.
- **Events:** `tts_chunk_requested`, `tts_first_audio`, `tts_audio` (the payload, never
  persisted), `tts_chunk_complete`, `tts_complete`.
- **Failures:** a backend that fails during synthesis is retried `streaming.retry_count`
  times. Then the next backend takes over and a `fallback` event is recorded, because the
  voice changes.

| Backend | Runs | License (engine / voice) |
|---|---|---|
| `piper` | local | GPL-3.0 engine / per-voice MODEL_CARD (default voice `en_US-libritts_r-medium`: CC BY 4.0, attribution required) |
| `sapi` | local, Windows | Windows built-in voices |
| `kokoro` | local | Apache-2.0 model; model files SHA-256 pinned |
| `mms` | local | CC-BY-NC-4.0 |
| `edge` | **cloud** | Microsoft online service terms. The text is sent to Microsoft, so this needs `fallback.allow_cloud` and a non-restricted data class. |

**Measured on the local CPU** (33 sentences: 8 with names and numbers, plus 25 FLORES English
references). "Intelligibility" is a Whisper `small` round trip (machine transcript of the
speech, scored against the text). It is a proxy, not a listening test.

| Voice | First audio p50 / p95 / p99 | RTF | Round-trip WER (all / names & numbers) | RAM |
|---|---|---|---|---|
| piper libritts_r | 218 / 452 / 504 ms | 0.063 | 0.074 / 0.210 | ~360 MB |
| piper lessac | 235 / 520 / 679 ms | 0.061 | 0.071 / 0.145 | ~400 MB |
| piper joe (CC0) | 278 / 522 / 603 ms | 0.064 | 0.092 / 0.237 | ~410 MB |
| sapi (Windows) | 35 / 57 / 66 ms | 0.006 | 0.056 / 0.105 | ~30 MB |
| kokoro af_heart (int8) | 6785 / 13119 / 17558 ms | 1.468 | 0.051 / 0.092 | ~510 MB |
| edge en-TZ-Imani (cloud, this network) | ~2.6–3.1 s per chunk | ~1.7 | not measured | — |

- **`cpu` and `local_windows` use `piper` `libritts_r`, with `sapi` as the fallback.** lessac
  is slightly more intelligible, but its dataset license is restricted.
- **Kokoro is the most intelligible but too slow on this CPU**, and Edge is slower than real
  time from this network. Both are opt-in.
- **Naturalness and prosody have not been rated yet.** They need human listeners (Phase 8).
- **East African names are the weak spot for every English voice.** "Suluhu Hassan" was heard
  as "Salu hasn't". A pronunciation dictionary is supported. The example
  `configs/pronunciation/sw_names_en.yaml` is **not** enabled by default: the Whisper round
  trip could not show a consistent gain, and capitalized respellings made things worse. Check
  entries by listening, ideally with a Swahili speaker.

## Full pipeline and Python API (Phase 6)

```powershell
python -m src.cli interpret --input audio.wav --profile cpu                  # batch, as fast as possible
python -m src.cli interpret --input audio.wav --profile cpu --live-sim --play  # real-time pace, speaker out
python -m src.cli interpret --mic --seconds 180 --play --profile cpu         # live microphone, turn-taking
python -m src.cli interpret --input audio.wav --no-tts --export              # text only + run directory
```

```python
from src.pipeline import RealtimeSwahiliInterpreter

async with RealtimeSwahiliInterpreter("cpu") as interp:       # start() ... shutdown()
    await interp.push_audio(chunk, sample_rate=16000)          # float32 or int16, any rate, mono/stereo
    async for event in interp.events():
        if event.type == "asr_final":  print("Swahili:", event.text)
        if event.type == "mt_final":   print("English:", event.text)
        if event.type == "tts_audio":  play(event.audio, event.data["sample_rate"])
```

The API also offers `flush()` (end the utterance and wait until it's spoken), `cancel()` (drop
queued work and stop speaking at the next chunk), `reset()` (cancel and forget context),
`status()`, `health()`, `model_status()`, `units()`, `summary()` and `shutdown()`.

**How the stages run:**
- **Three stage workers.** VAD+ASR, MT and TTS each run on their own single thread, so ASR
  can decode the next utterance while the previous one is translated and spoken.
- **Bounded queues.** Stages are connected by queues of `streaming.queue_max` items. A slow
  stage slows its input instead of growing memory. `push_audio(block=False)` drops and counts
  audio instead of waiting.
- **Backlog handling.** Audio that piles up while ASR is busy is processed in one pass, and
  ASR partials that a newer update already replaced are skipped.
- **Errors stay local.** A failed ASR decode loses only its own segment. A TTS failure after
  every fallback still delivers the text. Both emit an `error` event, and `health()` reports
  `degraded`.
- **Per-unit tracing.** Each translation unit ends with a `pipeline_complete` event carrying
  its latencies. `--export` writes `transcript_sw.txt`, `translation_en.txt`, `units.jsonl`,
  `speech_en.wav`, the events and a manifest. With `logging.log_text=false`, no text is
  written.
- **Playback metrics.** `--play` measures playback delay and underruns (gaps inside a unit).

**Measured on the local CPU** (`cpu` profile, `--live-sim`, 30 s synthetic Swahili clip, 6
utterances, one run per setting). Latency is from the last speech frame to the event.

| ASR partials | ASR final | MT final | First English audio p50 / p95 | RAM |
|---|---|---|---|---|
| on | 3.4 s | 4.0 s | 4.1 / 5.2 s | ~1.55 GB |
| **off (cpu default)** | 2.4 s | 2.8 s | **3.1 / 3.3 s** | ~1.55 GB |

- **Partials are off on CPU.** Each partial is another ~2 s Whisper `small` decode, and the
  end-of-utterance decode waits behind it. They are still available with
  `--set asr.partials=true` and are worth it on a GPU.
- **ASR dominates the remaining 3.1 s:** about 0.5 s waiting for silence, about 1.9 s of ASR
  decoding, about 0.5 s of MT and about 0.2 s for the first TTS chunk.
- **Translation quality was limited by Whisper `small`** on this CPU. For example, ASR heard
  "ya nyanya" (of tomatoes) as "yanyanya", which MT then rendered as "a bat". This is why the
  cpu profile now uses a Swahili fine-tune (see below; same 3.1 s first audio).
- **Batch latencies are not listener-relevant.** In batch mode audio arrives faster than real
  time, so the summary labels those numbers as including the backlog.

### Live microphone: turn-taking mode

`--mic --play` defaults to `--mode consecutive`, like a consecutive human interpreter:

1. A spoken "Ready. Please speak Swahili now." and a beep mark the start.
2. Say one sentence and pause. After 1.2 s of silence the turn ends (`vad.min_silence_ms=1200`
   in this mode, so short pauses inside a sentence don't cut it).
3. The mic is muted while the English is spoken, then a beep means "your turn".

- **Echo guard.** The mic is also muted while the speaker is playing, so the interpreter never
  hears and re-translates its own voice. `--no-echo-guard` turns it off (use with headphones).
- **Simultaneous mode** (`--mode simultaneous`) keeps listening while speaking. Use it only with
  headphones.
- `--no-cue` skips the spoken cue and beeps.
- Microphone audio is not saved. `--export` writes text, events and the English speech only.

### Swahili speech recognition model

OpenAI Whisper `small` is not good enough at Swahili. The cpu profile now uses a Swahili
fine-tune of `small` with the same speed. Any Hugging Face Whisper checkpoint can be set as
`asr.model`. It is converted once to CTranslate2 in `cache/ct2/asr/` (with a
`provenance.json`) by [src/asr/convert.py](src/asr/convert.py). The conversion needs
`transformers` and `torch` once; running it needs neither.

Model comparison measured with `scripts/asr_eval_fleurs.py` (beam 1, int8, normalized WER; FLEURS data from
`scripts/fetch_fleurs_sw.py`, CC-BY-4.0):

| Model | FLEURS sw test, 40 real utterances | Synthetic, 10 unseen clips | Decode p50 |
|---|---|---|---|
| Whisper `small` (OpenAI) | 0.791 | 0.734 | ~2.3 s |
| **Mollel/ASR-Swahili-Small (cpu default, Apache-2.0)** | **0.297** | **0.053** | ~2.3 s |
| cdli/whisper-small-swahili-afrivoice_cv (fallback) | 0.354 | 0.170 | ~2.3 s |
| ElizabethMwangi/whisper-small-swahili-afrivoice-cv | 0.360 | 0.138 | ~2.3 s |
| adoamesh/whisper-small-swh-finetuned (trained on FLEURS) | 0.290 | 0.192 | ~2.3 s |
| large-v3-turbo (OpenAI) | 0.349 | – | ~8.1 s |

**Beam search.** Mollel with beam 5 (the cpu default since 2026-11-23) has FLEURS WER 0.275
and synthetic WER 0.021, against 0.297 and 0.053 with beam 1. Each decode takes about 0.3 s
longer (p50 3.3 s vs 3.0 s). Use `--set asr.beam_size=1` for lower latency.

- Fine-tunes with their own custom vocabulary (e.g. alphaedge-ai) don't work with
  faster-whisper. It treats them as English-only.
- On Windows without Developer Mode, the Hugging Face cache can fail with WinError 1314
  (symlinks). The converter falls back to a plain-copy download folder.

**Live laptop-mic sessions** (2026-11-23, consecutive mode, beam 1, one speaker, 14 + 11 sentences):

- English started a median of 3.7 s and 3.9 s after the speaker stopped (range 3.2–5.0 s).
- There were no errors, no echo and no playback gaps.
- About half the sentences were translated correctly. Nearly all errors came from ASR, not MT:
  split or misheard formal words ("mwenye kiti" for "mwenyekiti", "madam barimbali" for
  "mambo mbalimbali"), and names.

### ASR correction list

[configs/asr_corrections/sw.csv](configs/asr_corrections/sw.csv) holds `heard,correct` pairs.
They rewrite recurring mishearings in the Swahili transcript before translation
(`asr.corrections_path`; `null` disables it). Matching is whole-word, case-insensitive and
longest-first. The uncorrected text is kept in the `asr_final` event as `original`, which is
dropped when `log_text=false`.

Add an entry only after the same mistake shows up more than once, and prefer phrases to
single words. The list only fixes mistakes it has already seen. Applied to the session it came
from, it turned "A gentleman with a chair" into "Mr. Chairman", and "the implementation …
has been successful" into "… would not have been successful". New speech is not helped until
its mistakes are added.

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
