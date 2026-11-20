"""Latency statistics. Never report only an average (spec section 12)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. Matches numpy's default method."""
    if not values:
        return math.nan
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Sequence[float]) -> dict[str, float | None]:
    """mean/median/p95/p99/min/max/std. Empty input gives None values (valid JSON, never NaN)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None,
                "p99": None, "min": None, "max": None, "std": None}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "std": math.sqrt(var),
    }
