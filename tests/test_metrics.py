"""Metrics layer: formula correctness + behavioral discrimination (FR-3.x / R-8)."""
import numpy as np
import pytest

from pabench.metrics import (METRIC_REGISTRY, auroc, band_power, dimensionless_jerk,
                             force_exceed_count, jerk_cmd, jitter_band_power,
                             plan_margin_ratio, success_rate, tracking_error,
                             uncertainty_failure_auroc, validate_registry, wilson_ci)


# ---------------- formula-level correctness


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    # boundaries
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo0, _ = wilson_ci(0, 20)
    assert lo0 == 0.0


def test_auroc_known_cases():
    assert auroc([1, 2, 3, 10, 11, 12], [False, False, False, True, True, True]) == 1.0
    assert auroc([10, 11, 12, 1, 2, 3], [False, False, False, True, True, True]) == 0.0
    assert auroc([5, 5, 5, 5], [True, False, True, False]) == 0.5  # all tied
    assert auroc([1, 2], [True, True]) is None  # single class → N/A


def test_dimensionless_jerk_orders_smoothness():
    t = np.arange(0, 2, 0.01)
    tau = t / t[-1]
    smooth = np.outer(10 * tau**3 - 15 * tau**4 + 6 * tau**5, [0.1, 0.0, 0.0])
    rng = np.random.default_rng(0)
    jerky = smooth + rng.normal(0, 1e-4, smooth.shape)
    assert dimensionless_jerk(t, jerky) > 10 * dimensionless_jerk(t, smooth)


def test_band_power_picks_in_band_signal():
    fs, n = 100.0, 800
    t = np.arange(n) / fs
    in_band = np.sin(2 * np.pi * 20.0 * t)    # 20 Hz ∈ [5,50]
    out_band = np.sin(2 * np.pi * 1.0 * t)    # 1 Hz ∉ [5,50]
    assert band_power(in_band, fs, 5, 50) > 100 * band_power(out_band, fs, 5, 50)


# ---------------- behavioral discrimination (metrics must distinguish good/bad models and good/bad hardware)


def test_sr_separates_models(precise_calibrated, sloppy_calibrated):
    sr_p = success_rate(precise_calibrated)["sr"]
    sr_s = success_rate(sloppy_calibrated)["sr"]
    assert sr_p > sr_s + 0.15, f"precise={sr_p:.2f} should be clearly higher than sloppy={sr_s:.2f}"


def test_sr_separates_hardware(precise_calibrated, precise_worn):
    assert success_rate(precise_calibrated)["sr"] > success_rate(precise_worn)["sr"]


def test_plan_margin_attributes_to_model_not_hw(precise_calibrated, precise_worn,
                                                sloppy_calibrated):
    """e_plan is a model quantity: swapping hardware should not change it noticeably; swapping models should (the FR-4.2 decomposition premise)."""
    m = lambda eps: np.mean([plan_margin_ratio(e) for e in eps])
    assert m(precise_calibrated) == pytest.approx(m(precise_worn), rel=0.05)
    assert m(sloppy_calibrated) > 2 * m(precise_calibrated)


def test_e_track_separates_hardware(precise_calibrated, precise_worn):
    """e_track is a hardware quantity: with the same model, swapping hardware should change it significantly (FR-3.12)."""
    t_cal = np.mean([tracking_error(e)["steady_rms_m"] for e in precise_calibrated])
    t_worn = np.mean([tracking_error(e)["steady_rms_m"] for e in precise_worn])
    assert t_worn > 3 * t_cal


def test_jitter_band_separates_hardware(precise_calibrated, precise_worn):
    """FR-3.11: the worn arm's 5–50 Hz jitter energy should be far higher than the calibrated arm's."""
    j_cal = np.mean([jitter_band_power(e) for e in precise_calibrated])
    j_worn = np.mean([jitter_band_power(e) for e in precise_worn])
    assert j_worn > 5 * j_cal


def test_jerk_cmd_separates_models(precise_calibrated, sloppy_calibrated):
    """FR-3.6: command jerk is attributed to the model — the jerky model should be far higher than the smooth one."""
    assert (np.median([jerk_cmd(e) for e in sloppy_calibrated])
            > 10 * np.median([jerk_cmd(e) for e in precise_calibrated]))


def test_uncertainty_auroc_calibrated_model(precise_calibrated):
    """FR-3.8a: well-calibrated uncertainty should predict failure (AUROC ≥ 0.7, R-3)."""
    v = uncertainty_failure_auroc(precise_calibrated)
    assert v is not None and v >= 0.7, f"AUROC={v}"


def test_force_exceed_higher_for_misaligned(precise_calibrated, sloppy_calibrated):
    """FR-3.7: a poorly-aligned model should have a higher total count of contact-force exceedances."""
    f = lambda eps: sum(force_exceed_count(e) for e in eps)
    assert f(sloppy_calibrated) > f(precise_calibrated)


# ---------------- registry machine-check (R-8)


def test_registry_validates():
    validate_registry()  # should not raise
    assert len(METRIC_REGISTRY) >= 8


def test_registry_rejects_metric_without_action():
    broken = {**METRIC_REGISTRY,
              "l9.orphan": {"level": "L9", "owner": "model",
                            "definition": "x", "improvement_actions": []}}
    with pytest.raises(ValueError, match="l9.orphan"):
        validate_registry(broken)