"""L0 结果层指标 (FR-3.1/3.2/3.4) —— 真实现。"""
from __future__ import annotations

import math
from collections import Counter

from ..schema import Episode


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间 (FR-3.1)。

    计算二项比例的 Wilson score 置信区间，相比朴素的正态近似
    在小样本或比例接近 0/1 时更稳健。

    Args:
        k: 成功次数。
        n: 总试验次数。
        z: 正态分布分位数，默认 1.96 对应 95% 置信水平。

    Returns:
        (下界, 上界)，均被裁剪到 [0, 1] 范围内。
    """
    # 无样本时无法估计，返回最宽区间 [0, 1]
    if n == 0:
        return (0.0, 1.0)
    p = k / n  # 样本比例（点估计）
    # Wilson 公式的公共分母：1 + z²/n
    denom = 1.0 + z * z / n
    # 区间中心：将点估计 p 向 1/2 收缩，修正小样本下的偏差
    center = (p + z * z / (2 * n)) / denom
    # 半宽：基于样本方差 p(1-p)/n 加上修正项 z²/(4n²)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # 裁剪到 [0, 1]，避免浮点误差导致越界
    return (max(0.0, center - half), min(1.0, center + half))


def success_rate(episodes: list[Episode]) -> dict:
    n = len(episodes)
    k = sum(1 for e in episodes if e.outcome.success)
    lo, hi = wilson_ci(k, n)
    return {"sr": k / n if n else float("nan"), "n": n, "successes": k, "ci95": (lo, hi)}


def efficiency_score(episodes: list[Episode], expert_duration_s: float) -> float | None:
    """FR-3.2: 成功回合时长 / 专家基准, 仅在成功回合上统计。"""
    durs = [e.outcome.duration_s for e in episodes if e.outcome.success]
    if not durs:
        return None
    return sum(durs) / len(durs) / expert_duration_s


def first_failure_histogram(episodes: list[Episode]) -> dict:
    """FR-3.4: 首次失败发生在哪个阶段的分布。"""
    c = Counter(e.outcome.failure_phase.value for e in episodes
                if not e.outcome.success and e.outcome.failure_phase)
    return dict(c)