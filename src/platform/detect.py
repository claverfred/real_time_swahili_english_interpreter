"""Detect which execution environment we are running in."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping


def is_colab(environ: Mapping[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in environ or "COLAB_GPU" in environ


def is_hpc(environ: Mapping[str, str] | None = None) -> bool:
    environ = os.environ if environ is None else environ
    return any(k in environ for k in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "PBS_JOBID"))


def detect_platform(environ: Mapping[str, str] | None = None) -> str:
    if is_colab(environ):
        return "colab"
    if is_hpc(environ):
        return "hpc"
    return "local"
