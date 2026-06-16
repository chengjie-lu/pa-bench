"""L2 process-quality metrics (FR-3.5–3.10) — real implementation, all computed from raw Episode telemetry (NFR-6)."""
from __future__ import annotations

import numpy as np

from ..schema import Episode, Phase


# ---------------------------------------------------------------- FR-3.5 alignment residual


def plan_margin_ratio(ep: Episode) -> float:
    """e_plan margin ratio: lateral deviation of the [commanded] position at the last pre-insertion step vs target ground truth / tolerance.
    >1 ⇒ the model's plan is already doomed to fail (physically impossible to succeed)."""
    _, _, ins_end = next(s for s in ep.robot.phase_spans if s[0] is Phase.INSERT)
    cmd_xy = ep.model.chunk.cmd_xyz[ins_end - 1, :2]
    target_xy = np.array(ep.scene.target_pose_gt.xyz[:2])
    return float(np.linalg.norm(cmd_xy - target_xy) / ep.scene.tolerance_class.gap_m)


# ---------------------------------------------------------------- FR-3.6 trajectory smoothness


def dimensionless_jerk(t: np.ndarray, pos: np.ndarray) -> float:
    """Dimensionless jerk integral: (T^5 / L^2) * ∫|x'''|^2 dt. Larger = less smooth."""
    dt = float(t[1] - t[0])
    j = np.diff(pos, n=3, axis=0) / dt**3
    integral = float(np.sum(j * j) * dt)
    T = float(t[-1] - t[0])
    L = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    if L <= 0:
        return 0.0
    return T**5 / L**2 * integral


def jerk_cmd(ep: Episode) -> float:
    """Computed on the command trajectory → attributed to the model."""
    return dimensionless_jerk(ep.model.chunk.t, ep.model.chunk.cmd_xyz)


def jerk_actual(ep: Episode) -> float:
    """Computed on the actual trajectory → attributed to the whole chain; its difference from jerk_cmd feeds attribution."""
    return dimensionless_jerk(ep.robot.t, ep.robot.ee_xyz_actual)


# ---------------------------------------------------------------- FR-3.7 force-interaction quality


def force_exceed_count(ep: Episode, f_max_n: float = 3.0) -> int:
    f = np.linalg.norm(ep.robot.ft_wrench[:, :3], axis=1)
    above = f > f_max_n
    # count "occurrences": number of rising edges
    return int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0))


# ---------------------------------------------------------------- FR-3.8 uncertainty


def peak_uncertainty(ep: Episode) -> float | None:
    ent = ep.model.chunk.entropy
    return None if ent is None else float(np.max(ent))


def auroc(scores, labels) -> float | None:
    """Mann-Whitney AUROC (with tie-averaged ranks), no third-party dependency. labels=True is the positive class (failure)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1
    ranks = np.empty(len(scores))
    ranks[order] = ranks_sorted
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def uncertainty_failure_auroc(episodes: list[Episode]) -> float | None:
    """How well peak uncertainty predicts failure (FR-3.8a). Returns None (N/A) when the model emits no uncertainty."""
    pairs = [(peak_uncertainty(e), not e.outcome.success) for e in episodes]
    pairs = [(s, y) for s, y in pairs if s is not None]
    if not pairs:
        return None
    return auroc([s for s, _ in pairs], [y for _, y in pairs])


# ---------------------------------------------------------------- FR-3.10 inference latency


def latency_percentiles(ep: Episode) -> dict:
    lat = ep.model.chunk.latency_ms
    return {"p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99))}