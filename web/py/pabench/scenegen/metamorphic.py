"""FR-1.3 metamorphic testing — this slice implements MR-1 (rotate the scene about the world z axis → actions should be equivariant).
MR-2/3/4 extend the same MetamorphicRelation interface (see rq.md FR-1.3, listed as a known limitation). Real implementation.
"""
from __future__ import annotations

import numpy as np

from ..schema import Episode, GenerationMethod, Scene


def dtw_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic O(N·M) DTW, returns optimal-path cost / max(N,M) — i.e. step-wise average deviation [m]."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Di, Dp, ci = D[i], D[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    return float(D[n, m] / max(n, m))


class MR1RotationZ:
    """MR-1: rotating the whole scene by theta about the world z axis ⇒ the model's command trajectory should rotate the same way (equivariance).
    Verdict: rotate the follow-up episode's command trajectory back into the source frame, run DTW against the source episode's command trajectory;
    step-wise average deviation > threshold_m ⇒ MR violated ⇒ model robustness defect (attribution directly assigns model).

    Note: equivariance only constrains the scene-anchored segment (from transfer onward — i.e. after leaving the fixed home);
    the approach segment starts from a home that never rotates with the scene, so it does not participate in the comparison.

    Distance metric: both trajectories are on the same time grid ⇒ use time-synchronized point-wise average distance (no warping loophole);
    DTW's time warping would absorb a constant bias along the path direction and weaken non-equivariance detection, so it is reserved for future
    unequal-length-trajectory scenarios (dtw_mean_distance is kept available).
    """
    mr_id = "MR-1"

    def __init__(self, theta_rad: float, threshold_m: float = 1.2e-3, dtw_stride: int = 8):
        self.theta = theta_rad
        self.threshold_m = threshold_m
        self.dtw_stride = dtw_stride

    def apply(self, base: Scene, parent_episode_id: str) -> Scene:
        return Scene(
            scene_id=f"{base.scene_id}-mr1-{self.theta:.2f}",
            task_type=base.task_type,
            tolerance_class=base.tolerance_class,
            part_pose_gt=base.part_pose_gt.rotated_z(self.theta),
            target_pose_gt=base.target_pose_gt.rotated_z(self.theta),
            perturbation={**base.perturbation, "mr1_theta": float(self.theta)},
            generation_method=GenerationMethod.METAMORPHIC,
            parent_episode_id=parent_episode_id,
            mr_id=self.mr_id,
        )

    def check(self, source_ep: Episode, followup_ep: Episode) -> dict:
        from ..schema import Phase
        i0, _ = source_ep.robot.span(Phase.TRANSFER)  # start of the scene-anchored segment
        src = source_ep.model.chunk.cmd_xyz[i0 :: self.dtw_stride]
        fol = followup_ep.model.chunk.cmd_xyz[i0 :: self.dtw_stride].copy()
        c, s = np.cos(-self.theta), np.sin(-self.theta)
        x, y = fol[:, 0].copy(), fol[:, 1].copy()
        fol[:, 0] = c * x - s * y
        fol[:, 1] = s * x + c * y
        d = float(np.mean(np.linalg.norm(src - fol, axis=1)))  # time-synchronized point-wise average distance
        return {"mr_id": self.mr_id, "theta": self.theta, "threshold_m": self.threshold_m,
                "mean_dist_m": d, "violated": d > self.threshold_m}


def mr_violation_verdict(checks: list[dict]) -> dict:
    """Protocol-level MR verdict (FR-1.3): take the median distance over multiple metamorphic follow-ups of the same source episode, then decide.
    A single comparison is swayed by the model's own random perception noise (false negative/positive); median aggregation improves test power."""
    if not checks:
        raise ValueError("at least 1 MR follow-up comparison is required")
    med = float(np.median([c["mean_dist_m"] for c in checks]))
    th = checks[0]["threshold_m"]
    return {"mr_id": checks[0]["mr_id"], "median_dist_m": med,
            "threshold_m": th, "violated": med > th, "n_followups": len(checks)}