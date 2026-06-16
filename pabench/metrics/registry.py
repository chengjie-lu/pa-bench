"""FR-5.1 metric registry — real implementation.
Machine-check rule (R-8): every metric must bind ≥1 improvement action, otherwise validate_registry raises and it cannot ship.
"""
from __future__ import annotations

METRIC_REGISTRY = {
    "l0.success_rate": {
        "level": "L0", "owner": "both",
        "definition": "successful episodes / total episodes, Wilson 95% CI (FR-3.1)",
        "improvement_actions": ["Drill down by task type × tolerance class × perturbation bucket, locate the weakest cell, then move to L1/L2 analysis"],
    },
    "l0.efficiency": {
        "level": "L0", "owner": "model",
        "definition": "successful-episode duration / expert baseline duration (FR-3.2)",
        "improvement_actions": ["Systematically high duration ⇒ check action-chunk step size and replanning frequency"],
    },
    "l0.first_failure_phase": {
        "level": "L1", "owner": "both",
        "definition": "distribution of the phase where the first failure occurs (FR-3.4)",
        "improvement_actions": ["Failures concentrated in grasp ⇒ add bin-grasping training data; concentrated in insert ⇒ inspect the L2 alignment residual"],
    },
    "l2.plan_margin_ratio": {
        "level": "L2", "owner": "model",
        "definition": "lateral offset of the pre-insertion commanded pose vs target ground truth / tolerance (FR-3.5)",
        "improvement_actions": ["Systematic offset ⇒ recheck hand-eye calibration / fine-tune vision; large random spread ⇒ raise perception resolution or add alignment data"],
    },
    "l2.jerk_cmd": {
        "level": "L2", "owner": "model",
        "definition": "dimensionless jerk integral of the commanded trajectory (FR-3.6)",
        "improvement_actions": ["High jerk ⇒ add action-smoothness regularization in training / low-pass filter at the output"],
    },
    "l2.force_exceed": {
        "level": "L2", "owner": "both",
        "definition": "number of times |F| exceeds the safety threshold during contact phases (FR-3.7)",
        "improvement_actions": ["Many exceedances ⇒ check alignment residual; if alignment is good yet still exceeding ⇒ check force-control params / contact model"],
    },
    "l2.uncertainty_auroc": {
        "level": "L2", "owner": "model",
        "definition": "AUROC of peak uncertainty for predicting failure (FR-3.8); N/A when the model emits no uncertainty",
        "improvement_actions": ["Low AUROC ⇒ calibration training of the uncertainty head; N/A ⇒ ask the model side to emit an action distribution"],
    },
    "l2.latency_p99": {
        "level": "L2", "owner": "model",
        "definition": "inference latency p99 (FR-3.10)",
        "improvement_actions": ["p99 exceeds the control period ⇒ distillation/quantization or asynchronous action chunks"],
    },
    "l3.e_track_rms": {
        "level": "L3", "owner": "hardware",
        "definition": "RMS of actual-minus-commanded SE(3) position deviation, align→fasten window (FR-3.12)",
        "improvement_actions": ["High RMS ⇒ tune controller gains; with a systematic offset ⇒ kinematic calibration"],
    },
    "l3.jitter_band_power": {
        "level": "L3", "owner": "hardware",
        "definition": "PSD energy of end-effector acceleration in the 5–50 Hz band (FR-3.11)",
        "improvement_actions": ["Energy exceeds baseline by 3σ ⇒ check gearbox wear / structural looseness, schedule maintenance"],
    },
}


def validate_registry(registry: dict = METRIC_REGISTRY) -> None:
    """R-8 machine-check: a metric with no improvement action bound is not allowed to ship."""
    offenders = [k for k, v in registry.items()
                 if not v.get("improvement_actions")]
    missing_def = [k for k, v in registry.items() if not v.get("definition")]
    if offenders or missing_def:
        raise ValueError(
            f"metric registry machine-check failed: missing improvement actions {offenders}; missing definitions {missing_def}")