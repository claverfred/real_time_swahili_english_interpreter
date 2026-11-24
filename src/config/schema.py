"""Typed configuration schema.

Every resolved configuration carries the fields required by the system spec (section 10):
environment, device, precision, model IDs + revisions, VAD thresholds, chunk/overlap sizes,
beam sizes, context window, token limits, buffers, queue limits, retries, timeouts, cache and
output paths, fallback policy, license status, logging, privacy, and retention.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Backends that send data to a third-party service. Used by the privacy guard.
CLOUD_TTS_BACKENDS = {"edge", "azure", "polly", "google", "elevenlabs"}
CLOUD_ASR_BACKENDS = {"azure", "google", "aws", "deepgram", "gemini"}
CLOUD_MT_BACKENDS = {"google", "azure", "deepl"}

Device = Literal["auto", "cpu", "cuda"]
Precision = Literal["auto", "fp32", "fp16", "bf16", "int8", "int8_float16"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EnvironmentConfig(_Strict):
    platform: Literal["auto", "local", "colab", "hpc"] = "auto"
    device: Device = "auto"
    gpu_index: int = Field(0, ge=0)
    precision: Precision = "auto"
    cpu_threads: int = Field(0, ge=0)
    seed: int = 1234


class AudioConfig(_Strict):
    sample_rate: int = Field(16000, gt=0)
    channels: int = Field(1, ge=1, le=2)
    frame_ms: Literal[10, 20, 30] = 30
    max_input_seconds: float = Field(3600, gt=0)
    max_input_mb: float = Field(500, gt=0)


class VadConfig(_Strict):
    backend: Literal["silero", "webrtc", "energy"] = "silero"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_speech_ms: int = Field(250, ge=0)
    min_silence_ms: int = Field(500, ge=0)
    speech_pad_ms: int = Field(200, ge=0)
    hangover_ms: int = Field(300, ge=0)
    max_utterance_s: float = Field(20.0, gt=0)
    silero_model_path: str | None = None   # default: <cache_dir>/models/silero_vad_v5.1.2/
    allow_model_download: bool = True      # one-time setup download; offline afterwards


class AsrConfig(_Strict):
    backend: str = "faster_whisper"
    model: str
    revision: str = "main"
    license: str = "unknown"
    language: str = "sw"
    task: Literal["transcribe"] = "transcribe"
    compute_type: str = "auto"
    beam_size: int = Field(5, ge=1, le=20)
    chunk_s: float = Field(1.0, gt=0)
    overlap_s: float = Field(0.5, ge=0)
    stable_prefix_agreement: int = Field(2, ge=1)
    partials: bool = True            # live partial hypotheses (each one is an extra decode)
    word_timestamps: bool = False
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    # Final-decode temperatures: when a decode fails the log-prob / compression checks, Whisper
    # retries at the next temperature (sampling). Each retry costs a full decode, so long lists
    # create multi-second latency spikes on CPU. Partials always use 0.0 only.
    temperatures: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    corrections_path: str | None = None   # Swahili->Swahili fixes for recurring ASR errors (CSV)
    fallback_models: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _overlap_lt_chunk(self) -> AsrConfig:
        if self.overlap_s >= self.chunk_s:
            raise ValueError(f"asr.overlap_s ({self.overlap_s}) must be < asr.chunk_s ({self.chunk_s})")
        return self


class MtConfig(_Strict):
    backend: Literal["nllb", "marian", "m2m", "madlad", "seamless"] = "nllb"
    model: str
    revision: str = "main"
    license: str = "unknown"
    src_lang: str = "swh_Latn"
    tgt_lang: str = "eng_Latn"
    beam_size: int = Field(4, ge=1, le=20)
    max_tokens: int = Field(256, gt=0)
    context_sentences: int = Field(2, ge=0)
    translate_partials: bool = False
    partial_min_new_words: int = Field(3, ge=1)
    engine: Literal["auto", "ctranslate2", "transformers"] = "auto"
    compute_type: str = "auto"
    context_mode: Literal["none", "prefix"] = "none"
    entity_strategy: Literal["verify", "mask"] = "verify"
    max_length_ratio: float = Field(3.0, gt=1.0)
    uncertain_policy: Literal["translate_flagged", "skip"] = "translate_flagged"
    glossary_path: str | None = None
    cache_size: int = Field(2048, ge=0)
    fallback_models: list[str] = Field(default_factory=list)


class TextConfig(_Strict):
    """Transcript stabilization and sentence aggregation before MT."""
    capitalize: bool = True
    add_final_punct: bool = True
    max_sentence_words: int = Field(40, ge=5)
    hold_incomplete_on_split: bool = True
    # Swahili number words -> digits before MT ("elfu mbili" -> 2000), so English keeps exact figures.
    number_words_to_digits: bool = False


class TtsConfig(_Strict):
    backend: str = "piper"
    model: str
    revision: str = "main"
    license: str = "unknown"
    voice: str
    sample_rate: int = Field(22050, gt=0)          # output rate; every backend is resampled to it
    chunk_max_chars: int = Field(220, ge=20)
    first_chunk_max_chars: int = Field(80, ge=20)  # short first chunk -> earlier first audio
    speaker_id: int | None = None                  # multi-speaker models (piper libritts_r, ...)
    speaking_rate: float = Field(1.0, gt=0.25, le=4.0)
    sentence_pause_ms: int = Field(120, ge=0)      # silence inserted between chunks
    fade_ms: float = Field(5.0, ge=0)              # fade in/out at chunk edges (no clicks)
    trim_silence: bool = True
    target_rms_dbfs: float | None = Field(-20.0, le=0)   # loudness normalization; null = off
    spell_acronyms: bool = True                    # "CCM" -> "C C M" unless in the dictionary
    pronunciation_dict: str | None = None
    voices: dict[str, str] = Field(default_factory=dict)  # voice per fallback backend
    voice_cloning: bool = False
    fallback_backends: list[str] = Field(default_factory=list)
    allow_model_download: bool = True

    @model_validator(mode="after")
    def _no_cloning(self) -> TtsConfig:
        if self.voice_cloning:
            raise ValueError(
                "tts.voice_cloning=true is refused by configuration: voice cloning requires "
                "explicit documented consent and is not enabled in this release."
            )
        return self


class StreamingConfig(_Strict):
    buffer_ms: int = Field(200, gt=0)
    queue_max: int = Field(64, gt=0)
    retry_count: int = Field(2, ge=0)
    timeout_s: float = Field(30.0, gt=0)


class FallbackConfig(_Strict):
    policy: Literal["allow_logged", "fail"] = "allow_logged"
    allow_cloud: bool = False


class StorageConfig(_Strict):
    cache_dir: str = "cache"
    output_dir: str = "results"
    log_dir: str = "logs"


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = Field(True, alias="json")
    log_text: bool = True
    log_audio: bool = False

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)


class PrivacyConfig(_Strict):
    data_class: Literal["public", "licensed", "restricted", "private"] = "public"
    allow_cloud_upload: bool = False
    retention_days: int = Field(30, ge=0)
    delete_temp_audio: bool = True
    session_timeout_min: int = Field(60, gt=0)


class ServerConfig(_Strict):
    """``serve``: local FastAPI/WebSocket server and browser UI."""
    host: str = "127.0.0.1"          # loopback only; any other host requires a token
    port: int = Field(8000, ge=1, le=65535)
    token_env: str = "SWI_SERVER_TOKEN"   # read the access token from this environment variable
    require_token: bool = False      # also on loopback (always on for other hosts)
    allowed_origins: list[str] = Field(default_factory=list)   # extra browser origins besides the server's own
    max_session_s: float = Field(1800.0, gt=0)      # a live session is stopped after this long
    idle_timeout_s: float = Field(120.0, gt=0)      # ... or after this long without any client message
    max_ws_message_bytes: int = Field(1_048_576, gt=0)   # one audio frame / control message
    rate_limit_per_min: int = Field(120, gt=0)      # HTTP API requests per client address
    preload: bool = True             # load models at startup (first session starts immediately)
    consecutive_silence_ms: int = Field(1200, ge=100)   # end-of-turn silence in turn-taking mode


class InterpreterConfig(_Strict):
    profile: str
    description: str = ""
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig
    text: TextConfig = Field(default_factory=TextConfig)
    mt: MtConfig
    tts: TtsConfig
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="after")
    def _privacy_guard(self) -> InterpreterConfig:
        """Refuse configurations that could send restricted data to a cloud service."""
        cloud_in_use = sorted(
            {self.tts.backend, *self.tts.fallback_backends} & CLOUD_TTS_BACKENDS
            | {self.asr.backend} & CLOUD_ASR_BACKENDS
        )
        restricted = self.privacy.data_class in {"restricted", "private"}
        if cloud_in_use and restricted:
            raise ValueError(
                f"privacy.data_class={self.privacy.data_class!r} forbids cloud backends, "
                f"but these are configured: {cloud_in_use}"
            )
        if cloud_in_use and not self.fallback.allow_cloud:
            raise ValueError(
                f"Cloud backends {cloud_in_use} are configured but fallback.allow_cloud is false. "
                "Enable it explicitly (e.g. the cloud_fallback or local_windows profile)."
            )
        if self.asr.backend in CLOUD_ASR_BACKENDS and not self.privacy.allow_cloud_upload:
            raise ValueError("A cloud ASR backend uploads audio; set privacy.allow_cloud_upload=true.")
        return self

    def uses_cloud(self) -> list[str]:
        return sorted(
            ({self.tts.backend, *self.tts.fallback_backends} & CLOUD_TTS_BACKENDS)
            | ({self.asr.backend} & CLOUD_ASR_BACKENDS)
        )
