from .detect import detect_platform, is_colab, is_hpc
from .resolve import FallbackRefused, RuntimeInfo, resolve_runtime

__all__ = ["FallbackRefused", "RuntimeInfo", "detect_platform", "is_colab", "is_hpc", "resolve_runtime"]
