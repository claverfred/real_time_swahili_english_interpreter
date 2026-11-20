"""CPU and memory introspection."""

from __future__ import annotations

import os
import platform

import psutil


def physical_cores() -> int:
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def auto_threads(requested: int = 0) -> int:
    """Threads for inference. Physical cores avoid SMT contention for INT8 GEMMs."""
    if requested > 0:
        return requested
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return max(1, physical_cores())


def cpu_info() -> dict[str, object]:
    vm = psutil.virtual_memory()
    return {
        "processor": platform.processor() or platform.machine(),
        "physical_cores": physical_cores(),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb": round(vm.available / 1e9, 2),
    }
