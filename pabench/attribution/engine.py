"""FR-4 故障归因引擎 —— 真实现 (规则决策树 + oracle 回放对照实验)。

决策树 (FR-4.1, 阈值版本化, rq.md O-6):
  1. MR 违反                          ⇒ model     (变质测试直接定责)
  2. e_plan 裕度比 > 1 且 e_track 正常 ⇒ model     (模型"想去的位置"就错了)
  3. e_plan 正常 且 e_track 超限       ⇒ hardware  (模型对了, 硬件没跟上)
  4. 其余 (两者都坏/都好) ⇒ oracle 回放对照 (FR-4.3):
       oracle 成功            ⇒ model       (换掉模型问题消失)
       oracle 失败 且光照 OOD  ⇒ environment  (完美计划也失败 + 扰动出训练分布)
       oracle 失败 且光照正常  ⇒ hardware
     无 oracle 可用           ⇒ ambiguous → 人工队列
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..schema import Attribution, Episode
from ..metrics.l2_process import plan_margin_ratio
from ..metrics.l3_hardware import tracking_error


@dataclass(frozen=True)
class AttributionThresholds:
    version: str = "attr-rules-0.1"
    plan_margin_fail: float = 1.0     # e_plan/公差 > 1 ⇒ 规划注定失败
    e_track_rms_max_m: float = 0.3e-3  # e_track RMS 超此值 ⇒ 硬件异常
    ood_lux_max: float = 0.4          # 光照低于此 ⇒ 视为出训练分布


def decide(mr_violated: bool, plan_ratio: float, e_track_rms_m: float,
           lux_factor: float, th: AttributionThresholds,
           oracle_success: Optional[bool] = None) -> tuple[Attribution, str]:
    """纯函数决策树 (便于单测与审计)。oracle_success=None 表示未做对照实验。"""
    plan_bad = plan_ratio > th.plan_margin_fail
    track_bad = e_track_rms_m > th.e_track_rms_max_m
    if mr_violated:
        return Attribution.MODEL, "rule1: MR violated ⇒ model robustness defect"
    if plan_bad and not track_bad:
        return Attribution.MODEL, f"rule2: e_plan margin ratio {plan_ratio:.2f}>1 and e_track normal"
    if not plan_bad and track_bad:
        return Attribution.HARDWARE, f"rule3: e_plan normal and e_track RMS {e_track_rms_m*1e3:.2f}mm over limit"
    if oracle_success is None:
        return Attribution.AMBIGUOUS, "rule4: no oracle control ⇒ route to manual queue"
    if oracle_success:
        return Attribution.MODEL, "rule4a: oracle succeeds under same conditions ⇒ model responsibility"
    if lux_factor < th.ood_lux_max:
        return Attribution.ENVIRONMENT, f"rule4b: oracle also fails and lux={lux_factor:.2f} out of training distribution"
    return Attribution.HARDWARE, "rule4c: oracle also fails and perturbation in distribution ⇒ hardware responsibility"


def attribute_episode(ep: Episode, th: AttributionThresholds,
                      mr_violated: bool = False,
                      oracle_fn: Optional[Callable[[Episode], bool]] = None) -> Optional[Attribution]:
    """对失败回合归因并写回 ep.outcome。成功回合返回 None。
    oracle_fn(ep)->bool: 触发一次 oracle 回放对照实验 (FR-2.5/FR-4.3), 仅 rule4 需要时调用。"""
    if ep.outcome.success:
        return None
    plan_ratio = plan_margin_ratio(ep)
    e_track = tracking_error(ep)["steady_rms_m"]  # 稳态窗口, 不受运动迟滞污染
    lux = float(ep.scene.perturbation.get("lux_factor", 1.0))
    plan_bad = plan_ratio > th.plan_margin_fail
    track_bad = e_track > th.e_track_rms_max_m
    oracle_success = None
    if not mr_violated and (plan_bad == track_bad) and oracle_fn is not None:
        oracle_success = oracle_fn(ep)  # 只在判不清时花对照实验预算 (rq.md D1)
    attribution, reason = decide(mr_violated, plan_ratio, e_track, lux, th, oracle_success)
    ep.outcome.attribution = attribution
    ep.outcome.attribution_reason = f"[{th.version}] {reason}"
    return attribution