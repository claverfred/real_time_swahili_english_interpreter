"""CUDA detection without forcing heavy imports when the libraries are absent."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int | None = None
    compute_capability: str | None = None


@dataclass
class CudaStatus:
    available: bool
    source: str                         # torch | ctranslate2 | nvidia-smi | none
    torch_version: str | None = None
    torch_cuda: str | None = None
    driver: str | None = None
    gpus: list[GpuInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def supports_bf16(self, index: int = 0) -> bool:
        if index >= len(self.gpus) or not self.gpus[index].compute_capability:
            return False
        return float(self.gpus[index].compute_capability) >= 8.0


def _nvidia_smi() -> tuple[str | None, list[GpuInfo]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None, []
    gpus, driver = [], None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            driver = parts[3]
            gpus.append(GpuInfo(int(parts[0]), parts[1], int(float(parts[2])), parts[4]))
    return driver, gpus


def probe_cuda() -> CudaStatus:
    driver, smi_gpus = _nvidia_smi()
    notes: list[str] = []

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    p = torch.cuda.get_device_properties(i)
                    gpus.append(GpuInfo(i, p.name, p.total_memory // (1024 * 1024), f"{p.major}.{p.minor}"))
                return CudaStatus(True, "torch", torch.__version__, torch.version.cuda, driver, gpus)
            notes.append(f"torch {torch.__version__} installed but CUDA unavailable "
                         f"(build CUDA: {torch.version.cuda or 'cpu-only build'})")
        except Exception as exc:  # broken torch install should not crash diagnosis
            notes.append(f"torch import failed: {exc}")

    if importlib.util.find_spec("ctranslate2") is not None:
        try:
            import ctranslate2

            n = ctranslate2.get_cuda_device_count()
            if n > 0:
                return CudaStatus(True, "ctranslate2", driver=driver, gpus=smi_gpus, notes=notes)
        except Exception as exc:
            notes.append(f"ctranslate2 CUDA probe failed: {exc}")

    if smi_gpus:
        notes.append("nvidia-smi sees a GPU but no CUDA-enabled inference library is installed")
        return CudaStatus(False, "nvidia-smi", driver=driver, gpus=smi_gpus, notes=notes)

    notes.append("no NVIDIA GPU detected")
    return CudaStatus(False, "none", notes=notes)
