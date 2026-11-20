from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MtResult:
    text: str
    source: str
    model: str
    engine: str
    latency_ms: float
    cached: bool = False
    provisional: bool = False        # translation of a partial (stable-prefix) transcript
    uncertain: bool = False          # source transcript was flagged uncertain by ASR
    context_used: int = 0            # previous sentences given as context
    flags: list[str] = field(default_factory=list)
    preservation: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
