"""FR-1.1 标称任务集 —— 本纵切只落地 1 个任务: 瓶盖拧紧 screw_cap @ T1 (rq.md §11 M1)。
其余 4 个任务型按同一 Scene 契约扩展, 不需要改 schema (NFR-5)。真实现。"""
from __future__ import annotations

from ..schema import GenerationMethod, Phase, SE3Pose, Scene, TaskType, ToleranceClass

# 阶段计划: (Phase, 时长 s) — Runner 据此切分 phase_spans (FR-3.4 的规则切分基准)
PHASE_PLAN = [
    (Phase.APPROACH, 1.5),
    (Phase.GRASP, 0.5),
    (Phase.TRANSFER, 2.0),
    (Phase.ALIGN, 1.5),
    (Phase.INSERT, 1.0),
    (Phase.FASTEN, 1.5),
]
EXPERT_DURATION_S = sum(d for _, d in PHASE_PLAN)  # FR-3.2 效率分的专家基准时长


def nominal_screw_cap(tolerance: ToleranceClass = ToleranceClass.T1) -> Scene:
    """料箱取瓶盖 → 拧紧到瓶身, 标称场景 (无扰动)。"""
    return Scene(
        scene_id=f"screw_cap-{tolerance.value}-nominal",
        task_type=TaskType.SCREW_CAP,
        tolerance_class=tolerance,
        part_pose_gt=SE3Pose((0.40, -0.20, 0.05), 0.0),   # 瓶盖在料箱中
        target_pose_gt=SE3Pose((0.10, 0.30, 0.15), 0.0),  # 瓶口
        perturbation={},
        generation_method=GenerationMethod.NOMINAL,
    )