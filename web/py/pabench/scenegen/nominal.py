"""FR-1.1 nominal task set — this slice implements only 1 task: cap fastening screw_cap @ T1 (rq.md §11 M1).
The other 4 task types extend the same Scene contract with no schema change (NFR-5). Real implementation."""
from __future__ import annotations

from ..schema import GenerationMethod, Phase, SE3Pose, Scene, TaskType, ToleranceClass

# Phase plan: (Phase, duration s) — the Runner uses this to segment phase_spans (the rule-segmentation basis of FR-3.4)
PHASE_PLAN = [
    (Phase.APPROACH, 1.5),
    (Phase.GRASP, 0.5),
    (Phase.TRANSFER, 2.0),
    (Phase.ALIGN, 1.5),
    (Phase.INSERT, 1.0),
    (Phase.FASTEN, 1.5),
]
EXPERT_DURATION_S = sum(d for _, d in PHASE_PLAN)  # FR-3.2 expert baseline duration for the efficiency score


def nominal_screw_cap(tolerance: ToleranceClass = ToleranceClass.T1) -> Scene:
    """Pick the cap from the bin → screw it onto the bottle, nominal scene (no perturbation)."""
    return Scene(
        scene_id=f"screw_cap-{tolerance.value}-nominal",
        task_type=TaskType.SCREW_CAP,
        tolerance_class=tolerance,
        part_pose_gt=SE3Pose((0.40, -0.20, 0.05), 0.0),   # cap in the bin
        target_pose_gt=SE3Pose((0.10, 0.30, 0.15), 0.0),  # bottle mouth
        perturbation={},
        generation_method=GenerationMethod.NOMINAL,
    )