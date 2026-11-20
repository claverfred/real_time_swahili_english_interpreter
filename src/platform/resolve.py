"""Resolve "auto" platform/device/precision values into concrete settings.

Order (spec section 8): explicit CLI/config -> environment -> profile -> CUDA auto-detect -> CPU.
Every fallback is recorded in the :class:`FallbackLog`; with ``fallback.policy == "fail"`` a
fallback raises instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config.schema import InterpreterConfig
from ..utils.manifest import FallbackLog
from .cpu import auto_threads
from .cuda import CudaStatus, probe_cuda
from .detect import detect_platform

_CT2_COMPUTE = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "int8": "int8",
    "int8_float16": "int8_float16",
}
_GPU_ONLY_PRECISIONS = {"fp16", "bf16", "int8_float16"}


class FallbackRefused(RuntimeError):
    """A fallback was required but ``fallback.policy`` is ``fail``."""


@dataclass
class RuntimeInfo:
    platform: str
    device: str
    precision: str
    asr_compute_type: str
    cpu_threads: int
    gpu_name: str | None
    gpu_memory_mb: int | None
    cuda_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_runtime(
    cfg: InterpreterConfig,
    fallbacks: FallbackLog | None = None,
    cuda: CudaStatus | None = None,
) -> tuple[InterpreterConfig, RuntimeInfo, FallbackLog]:
    fallbacks = fallbacks or FallbackLog()
    cfg = cfg.model_copy(deep=True)
    env = cfg.environment
    strict = cfg.fallback.policy == "fail"

    def fallback(stage: str, requested: str, selected: str, reason: str) -> None:
        if strict:
            raise FallbackRefused(f"{stage}: {requested} unavailable ({reason}); fallback.policy=fail")
        fallbacks.add(stage, requested, selected, reason)

    if env.platform == "auto":
        env.platform = detect_platform()

    cuda = cuda or probe_cuda()
    if env.device == "auto":
        env.device = "cuda" if cuda.available else "cpu"
    elif env.device == "cuda" and not cuda.available:
        fallback("device", "cuda", "cpu", "; ".join(cuda.notes) or "CUDA unavailable")
        env.device = "cpu"
    if env.device == "cuda" and env.gpu_index >= len(cuda.gpus) and cuda.gpus:
        fallback("device", f"cuda:{env.gpu_index}", "cuda:0", f"only {len(cuda.gpus)} GPU(s) visible")
        env.gpu_index = 0

    if env.precision == "auto":
        if env.device == "cpu":
            env.precision = "int8"
        else:
            env.precision = "bf16" if cuda.supports_bf16(env.gpu_index) else "fp16"
    elif env.device == "cpu" and env.precision in _GPU_ONLY_PRECISIONS:
        fallback("precision", env.precision, "int8", "precision not supported on CPU")
        env.precision = "int8"
    elif env.precision == "bf16" and env.device == "cuda" and not cuda.supports_bf16(env.gpu_index):
        fallback("precision", "bf16", "fp16", "GPU compute capability < 8.0")
        env.precision = "fp16"

    compute = cfg.asr.compute_type
    if compute == "auto":
        compute = _CT2_COMPUTE[env.precision]
    elif env.device == "cpu" and compute in {"float16", "bfloat16", "int8_float16", "int8_bfloat16"}:
        fallback("asr.compute_type", compute, "int8", "compute type not supported on CPU")
        compute = "int8"
    cfg.asr.compute_type = compute

    env.cpu_threads = auto_threads(env.cpu_threads)

    gpu = cuda.gpus[env.gpu_index] if env.device == "cuda" and cuda.gpus else None
    info = RuntimeInfo(
        platform=env.platform,
        device=env.device,
        precision=env.precision,
        asr_compute_type=compute,
        cpu_threads=env.cpu_threads,
        gpu_name=gpu.name if gpu else None,
        gpu_memory_mb=gpu.memory_total_mb if gpu else None,
        cuda_source=cuda.source,
    )
    return cfg, info, fallbacks
