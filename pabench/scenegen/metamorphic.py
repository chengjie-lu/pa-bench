"""FR-1.3 变质测试 —— 本纵切落地 MR-1 (场景绕世界 z 轴旋转 → 动作应等变)。
MR-2/3/4 按同一 MetamorphicRelation 接口扩展 (见 rq.md FR-1.3, 列入已知限制)。真实现。
"""
from __future__ import annotations

import numpy as np

from ..schema import Episode, GenerationMethod, Scene


def dtw_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """经典 O(N·M) DTW, 返回最优路径代价 / max(N,M) — 即逐步平均偏差 [m]。"""
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
    """MR-1: 整个场景绕世界 z 轴旋转 theta ⇒ 模型指令轨迹应近似同旋转 (等变性)。
    判定: 把后继回合的指令轨迹反旋回源坐标系, 与源回合指令轨迹做 DTW;
    逐步平均偏差 > threshold_m ⇒ MR 违反 ⇒ 模型鲁棒性缺陷 (归因直接判 model)。

    注意: 等变性只约束场景锚定段 (transfer 起 — 即离开固定 home 之后);
    approach 段从不随场景旋转的 home 出发, 不参与比较。

    距离度量: 两轨迹在同一时间网格上 ⇒ 用时间同步逐点平均距离 (无规整漏洞);
    DTW 的时间规整会沿路径方向吸收恒定偏置, 弱化非等变检出, 仅留给将来
    不等长轨迹的场景 (dtw_mean_distance 保留可用)。
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
        i0, _ = source_ep.robot.span(Phase.TRANSFER)  # 场景锚定段起点
        src = source_ep.model.chunk.cmd_xyz[i0 :: self.dtw_stride]
        fol = followup_ep.model.chunk.cmd_xyz[i0 :: self.dtw_stride].copy()
        c, s = np.cos(-self.theta), np.sin(-self.theta)
        x, y = fol[:, 0].copy(), fol[:, 1].copy()
        fol[:, 0] = c * x - s * y
        fol[:, 1] = s * x + c * y
        d = float(np.mean(np.linalg.norm(src - fol, axis=1)))  # 时间同步逐点平均距离
        return {"mr_id": self.mr_id, "theta": self.theta, "threshold_m": self.threshold_m,
                "mean_dist_m": d, "violated": d > self.threshold_m}


def mr_violation_verdict(checks: list[dict]) -> dict:
    """协议级 MR 判定 (FR-1.3): 对同一源回合的多个变质后继取中位距离再判。
    单次比较会被模型自身的随机感知噪声左右 (假阴/假阳), 中位数聚合提高检验功效。"""
    if not checks:
        raise ValueError("至少需要 1 个 MR 后继比较")
    med = float(np.median([c["mean_dist_m"] for c in checks]))
    th = checks[0]["threshold_m"]
    return {"mr_id": checks[0]["mr_id"], "median_dist_m": med,
            "threshold_m": th, "violated": med > th, "n_followups": len(checks)}