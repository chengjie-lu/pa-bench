"""L0 outcome-layer metrics (FR-3.1/3.2/3.4) — real implementation."""
from __future__ import annotations

import math
from collections import Counter

from ..schema import Episode


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (FR-3.1).

    Computes the Wilson score confidence interval for a binomial proportion;
    compared with the naive normal approximation it is more robust for small
    samples or proportions close to 0/1.

    Args:
        k: number of successes.
        n: total number of trials.
        z: normal-distribution quantile, default 1.96 for the 95% confidence level.

    Returns:
        (lower bound, upper bound), both clamped to the [0, 1] range.
    """
    # with no samples it cannot be estimated, return the widest interval [0, 1]
    if n == 0:
        return (0.0, 1.0)
    p = k / n  # sample proportion (point estimate)
    # common denominator of the Wilson formula: 1 + z²/n
    denom = 1.0 + z * z / n
    # interval center: shrink the point estimate p toward 1/2, correcting small-sample bias
    center = (p + z * z / (2 * n)) / denom
    # half width: based on sample variance p(1-p)/n plus the correction term z²/(4n²)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # clamp to [0, 1] to avoid floating-point overshoot
    return (max(0.0, center - half), min(1.0, center + half))


def success_rate(episodes: list[Episode]) -> dict:
    n = len(episodes)
    k = sum(1 for e in episodes if e.outcome.success)
    lo, hi = wilson_ci(k, n)
    return {"sr": k / n if n else float("nan"), "n": n, "successes": k, "ci95": (lo, hi)}


def efficiency_score(episodes: list[Episode], expert_duration_s: float) -> float | None:
    """FR-3.2: successful-episode duration / expert baseline, computed over successful episodes only."""
    durs = [e.outcome.duration_s for e in episodes if e.outcome.success]
    if not durs:
        return None
    return sum(durs) / len(durs) / expert_duration_s


def first_failure_histogram(episodes: list[Episode]) -> dict:
    """FR-3.4: distribution of which phase the first failure occurs in."""
    c = Counter(e.outcome.failure_phase.value for e in episodes
                if not e.outcome.success and e.outcome.failure_phase)
    return dict(c)