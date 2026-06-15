"""L2 过程质量层指标 (FR-3.5–3.10) —— 真实现, 全部从 Episode 原始遥测计算 (NFR-6)。"""
from __future__ import annotations

import numpy as np

from ..schema import Episode, Phase


# ---------------------------------------------------------------- FR-3.5 对准残差


def plan_margin_ratio(ep: Episode) -> float:
    """e_plan 裕度比: 插入前最后时刻【指令】位置相对目标真值的横向偏差 / 公差。
    >1 ⇒ 模型规划已注定失败 (物理上不可能成功)。"""
    _, _, ins_end = next(s for s in ep.robot.phase_spans if s[0] is Phase.INSERT)
    cmd_xy = ep.model.chunk.cmd_xyz[ins_end - 1, :2]
    target_xy = np.array(ep.scene.target_pose_gt.xyz[:2])
    return float(np.linalg.norm(cmd_xy - target_xy) / ep.scene.tolerance_class.gap_m)


# ---------------------------------------------------------------- FR-3.6 轨迹平滑度


def dimensionless_jerk(t: np.ndarray, pos: np.ndarray) -> float:
    """无量纲 jerk 积分: (T^5 / L^2) * ∫|x'''|^2 dt。值越大越不平滑。"""
    dt = float(t[1] - t[0])
    j = np.diff(pos, n=3, axis=0) / dt**3
    integral = float(np.sum(j * j) * dt)
    T = float(t[-1] - t[0])
    L = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    if L <= 0:
        return 0.0
    return T**5 / L**2 * integral


def jerk_cmd(ep: Episode) -> float:
    """对指令轨迹计算 → 归模型。"""
    return dimensionless_jerk(ep.model.chunk.t, ep.model.chunk.cmd_xyz)


def jerk_actual(ep: Episode) -> float:
    """对实测轨迹计算 → 归整链路; 与 jerk_cmd 的差参与归因。"""
    return dimensionless_jerk(ep.robot.t, ep.robot.ee_xyz_actual)


# ---------------------------------------------------------------- FR-3.7 力交互质量


def force_exceed_count(ep: Episode, f_max_n: float = 3.0) -> int:
    f = np.linalg.norm(ep.robot.ft_wrench[:, :3], axis=1)
    above = f > f_max_n
    # 计"次数": 上升沿个数
    return int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0))


# ---------------------------------------------------------------- FR-3.8 不确定性


def peak_uncertainty(ep: Episode) -> float | None:
    ent = ep.model.chunk.entropy
    return None if ent is None else float(np.max(ent))


def auroc(scores, labels) -> float | None:
    """Mann-Whitney AUROC (带并列秩平均), 无第三方依赖。labels=True 为正类(失败)。"""
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
    """峰值不确定性对失败的预警能力 (FR-3.8a)。模型不输出不确定性时返回 None (N/A)。"""
    pairs = [(peak_uncertainty(e), not e.outcome.success) for e in episodes]
    pairs = [(s, y) for s, y in pairs if s is not None]
    if not pairs:
        return None
    return auroc([s for s, _ in pairs], [y for _, y in pairs])


# ---------------------------------------------------------------- FR-3.10 推理时延


def latency_percentiles(ep: Episode) -> dict:
    lat = ep.model.chunk.latency_ms
    return {"p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99))}