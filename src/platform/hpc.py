"""Thin HPC helpers. Reads scheduler state only; never invents cluster details."""

from __future__ import annotations

import os

_SLURM_VARS = [
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
    "SLURM_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_GPUS", "SLURM_JOB_GPUS",
    "SLURM_MEM_PER_NODE", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT",
]


def slurm_info() -> dict[str, str]:
    """Scheduler context of the current job (empty outside Slurm)."""
    return {k: os.environ[k] for k in _SLURM_VARS if k in os.environ}


def is_login_node() -> bool:
    """Heuristic: on a cluster but not inside a job allocation."""
    return "SLURM_JOB_ID" not in os.environ and bool(os.environ.get("SLURM_CONF") or os.path.exists("/etc/slurm"))
