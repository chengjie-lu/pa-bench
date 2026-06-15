"""指标层: 公式正确性 + 行为区分度 (FR-3.x / R-8)。"""
import numpy as np
import pytest

from pabench.metrics import (METRIC_REGISTRY, auroc, band_power, dimensionless_jerk,
                             force_exceed_count, jerk_cmd, jitter_band_power,
                             plan_margin_ratio, success_rate, tracking_error,
                             uncertainty_failure_auroc, validate_registry, wilson_ci)


# ---------------- 公式级正确性


def test_wilson_ci_known_value():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    # 边界
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo0, _ = wilson_ci(0, 20)
    assert lo0 == 0.0


def test_auroc_known_cases():
    assert auroc([1, 2, 3, 10, 11, 12], [False, False, False, True, True, True]) == 1.0
    assert auroc([10, 11, 12, 1, 2, 3], [False, False, False, True, True, True]) == 0.0
    assert auroc([5, 5, 5, 5], [True, False, True, False]) == 0.5  # 全并列
    assert auroc([1, 2], [True, True]) is None  # 单类 → N/A


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


# ---------------- 行为区分度 (指标必须能区分好/坏模型与好/坏硬件)


def test_sr_separates_models(precise_calibrated, sloppy_calibrated):
    sr_p = success_rate(precise_calibrated)["sr"]
    sr_s = success_rate(sloppy_calibrated)["sr"]
    assert sr_p > sr_s + 0.15, f"precise={sr_p:.2f} 应明显高于 sloppy={sr_s:.2f}"


def test_sr_separates_hardware(precise_calibrated, precise_worn):
    assert success_rate(precise_calibrated)["sr"] > success_rate(precise_worn)["sr"]


def test_plan_margin_attributes_to_model_not_hw(precise_calibrated, precise_worn,
                                                sloppy_calibrated):
    """e_plan 是模型量: 换硬件不应明显改变; 换模型应明显改变 (FR-4.2 分解前提)。"""
    m = lambda eps: np.mean([plan_margin_ratio(e) for e in eps])
    assert m(precise_calibrated) == pytest.approx(m(precise_worn), rel=0.05)
    assert m(sloppy_calibrated) > 2 * m(precise_calibrated)


def test_e_track_separates_hardware(precise_calibrated, precise_worn):
    """e_track 是硬件量: 同模型换硬件应显著变化 (FR-3.12)。"""
    t_cal = np.mean([tracking_error(e)["steady_rms_m"] for e in precise_calibrated])
    t_worn = np.mean([tracking_error(e)["steady_rms_m"] for e in precise_worn])
    assert t_worn > 3 * t_cal


def test_jitter_band_separates_hardware(precise_calibrated, precise_worn):
    """FR-3.11: 磨损臂 5–50 Hz 抖动能量应远高于校准臂。"""
    j_cal = np.mean([jitter_band_power(e) for e in precise_calibrated])
    j_worn = np.mean([jitter_band_power(e) for e in precise_worn])
    assert j_worn > 5 * j_cal


def test_jerk_cmd_separates_models(precise_calibrated, sloppy_calibrated):
    """FR-3.6: 指令 jerk 归模型 — 抖动模型应远高于平滑模型。"""
    assert (np.median([jerk_cmd(e) for e in sloppy_calibrated])
            > 10 * np.median([jerk_cmd(e) for e in precise_calibrated]))


def test_uncertainty_auroc_calibrated_model(precise_calibrated):
    """FR-3.8a: 校准良好的不确定性应能预警失败 (AUROC ≥ 0.7, R-3)。"""
    v = uncertainty_failure_auroc(precise_calibrated)
    assert v is not None and v >= 0.7, f"AUROC={v}"


def test_force_exceed_higher_for_misaligned(precise_calibrated, sloppy_calibrated):
    """FR-3.7: 对准差的模型接触力超限次数总和应更多。"""
    f = lambda eps: sum(force_exceed_count(e) for e in eps)
    assert f(sloppy_calibrated) > f(precise_calibrated)


# ---------------- 注册表机检 (R-8)


def test_registry_validates():
    validate_registry()  # 不应抛错
    assert len(METRIC_REGISTRY) >= 8


def test_registry_rejects_metric_without_action():
    broken = {**METRIC_REGISTRY,
              "l9.orphan": {"level": "L9", "owner": "model",
                            "definition": "x", "improvement_actions": []}}
    with pytest.raises(ValueError, match="l9.orphan"):
        validate_registry(broken)