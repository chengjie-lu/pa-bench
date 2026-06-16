"""FR-4 failure-attribution engine — real implementation (rule decision tree + oracle-replay control experiment).

Decision tree (FR-4.1, versioned thresholds, rq.md O-6):
  1. MR violated                          ⇒ model     (the metamorphic test assigns blame directly)
  2. e_plan margin ratio > 1 and e_track normal ⇒ model (the position the model "wanted to reach" was already wrong)
  3. e_plan normal and e_track over limit  ⇒ hardware  (the model was right, the hardware did not keep up)
  4. otherwise (both bad / both good) ⇒ oracle-replay control (FR-4.3):
       oracle succeeds          ⇒ model        (swapping the model makes the problem go away)
       oracle fails and lighting OOD ⇒ environment (even a perfect plan fails + perturbation outside the training distribution)
       oracle fails and lighting normal ⇒ hardware
     no oracle available        ⇒ ambiguous → manual queue
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
    plan_margin_fail: float = 1.0     # e_plan/tolerance > 1 ⇒ plan is doomed to fail
    e_track_rms_max_m: float = 0.3e-3  # e_track RMS above this ⇒ hardware anomaly
    ood_lux_max: float = 0.4          # lighting below this ⇒ treated as out-of-distribution


def decide(mr_violated: bool, plan_ratio: float, e_track_rms_m: float,
           lux_factor: float, th: AttributionThresholds,
           oracle_success: Optional[bool] = None) -> tuple[Attribution, str]:
    """Pure-function decision tree (easy to unit-test and audit). oracle_success=None means no control experiment was run."""
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
    """Attribute a failed episode and write back to ep.outcome. Returns None for successful episodes.
    oracle_fn(ep)->bool: triggers one oracle-replay control experiment (FR-2.5/FR-4.3), called only when rule4 needs it."""
    if ep.outcome.success:
        return None
    plan_ratio = plan_margin_ratio(ep)
    e_track = tracking_error(ep)["steady_rms_m"]  # steady-state window, not contaminated by motion lag
    lux = float(ep.scene.perturbation.get("lux_factor", 1.0))
    plan_bad = plan_ratio > th.plan_margin_fail
    track_bad = e_track > th.e_track_rms_max_m
    oracle_success = None
    if not mr_violated and (plan_bad == track_bad) and oracle_fn is not None:
        oracle_success = oracle_fn(ep)  # spend control-experiment budget only when inconclusive (rq.md D1)
    attribution, reason = decide(mr_violated, plan_ratio, e_track, lux, th, oracle_success)
    ep.outcome.attribution = attribution
    ep.outcome.attribution_reason = f"[{th.version}] {reason}"
    return attribution