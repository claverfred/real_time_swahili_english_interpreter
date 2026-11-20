"""Serializable ASR result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AsrWord:
    text: str
    start: float
    end: float
    probability: float


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float = 0.0
    words: list[AsrWord] = field(default_factory=list)
    uncertain: bool = False
    rejected: str | None = None      # reason the filter dropped it; None = kept


@dataclass
class AsrResult:
    text: str
    language: str
    language_forced: bool
    duration_s: float
    inference_ms: float
    model: str
    segments: list[AsrSegment] = field(default_factory=list)

    @property
    def kept_segments(self) -> list[AsrSegment]:
        return [s for s in self.segments if s.rejected is None]

    @property
    def rejected_segments(self) -> list[AsrSegment]:
        return [s for s in self.segments if s.rejected is not None]

    @property
    def uncertain(self) -> bool:
        return any(s.uncertain for s in self.kept_segments)

    @property
    def rtf(self) -> float:
        return (self.inference_ms / 1000.0) / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def words(self) -> list[AsrWord]:
        return [w for s in self.kept_segments for w in s.words]

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d.update(rtf=round(self.rtf, 4), uncertain=self.uncertain)
        return d
