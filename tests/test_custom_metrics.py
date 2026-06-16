"""User-registered metrics (FR-5.1 extension): safe formula evaluation, aggregation, and persistence."""
import pytest

from pabench.metrics import (compile_expr, compute_for_combos, evaluate, validate_spec,
                             MetricSpecError, CustomMetricStore)


RECS = [
    {"model_id": "m", "hw_config_id": "h", "success": True, "plan_margin_ratio": 0.5,
     "e_track_steady_rms_mm": 0.6, "peak_uncertainty": 1.2, "lux": 0.5, "duration_s": 8.0},
    {"model_id": "m", "hw_config_id": "h", "success": False, "plan_margin_ratio": 1.3,
     "e_track_steady_rms_mm": 0.2, "peak_uncertainty": None, "lux": 0.3, "duration_s": 9.0},
]


def _spec(expr, agg="mean", mid="custom.t"):
    return validate_spec({"metric_id": mid, "level": "L2", "owner": "model",
                          "definition": "d", "expr": expr, "agg": agg,
                          "improvement_actions": ["do something"]})


# ---------------- safe evaluation + aggregation ----------------

def test_mean_of_field():
    assert evaluate(_spec("plan_margin_ratio", "mean"), RECS) == pytest.approx(0.9)


def test_boolean_predicate_rate():
    # e_track > 0.5: rec0 True, rec1 False → 0.5
    assert evaluate(_spec("e_track_steady_rms_mm > 0.5", "rate"), RECS) == pytest.approx(0.5)


def test_success_coerced_to_number():
    assert evaluate(_spec("1 - success", "mean"), RECS) == pytest.approx(0.5)  # failure rate


@pytest.mark.parametrize("agg,expected", [
    ("max", 1.3), ("min", 0.5), ("sum", 1.8), ("median", 0.9)])
def test_aggregations(agg, expected):
    assert evaluate(_spec("plan_margin_ratio", agg), RECS) == pytest.approx(expected)


def test_records_with_missing_field_are_skipped():
    # peak_uncertainty is None on rec1 → only rec0 (1.2) contributes
    assert evaluate(_spec("peak_uncertainty", "mean"), RECS) == pytest.approx(1.2)


def test_all_missing_returns_none():
    recs = [{"model_id": "m", "hw_config_id": "h", "peak_uncertainty": None}]
    assert evaluate(_spec("peak_uncertainty", "mean"), recs) is None


def test_functions_allowed():
    assert evaluate(_spec("min(plan_margin_ratio, 1.0)", "max"), RECS) == pytest.approx(1.0)


def test_compute_for_combos_groups_by_combo():
    recs = RECS + [{"model_id": "m2", "hw_config_id": "h", "success": True,
                    "plan_margin_ratio": 0.1, "e_track_steady_rms_mm": 0.1,
                    "peak_uncertainty": 0.1, "lux": 0.9, "duration_s": 7.0}]
    out = compute_for_combos([_spec("plan_margin_ratio", "mean")], recs)
    assert out["m @ h"]["custom.t"] == pytest.approx(0.9)
    assert out["m2 @ h"]["custom.t"] == pytest.approx(0.1)


# ---------------- safety: unsafe formulas are rejected ----------------

@pytest.mark.parametrize("bad", [
    "__import__('os').system('x')",
    "open('/etc/passwd')",
    "success.__class__",
    "lux if True else exec('1')",
    "unknown_field + 1",
    "[x for x in (1,2)]",
    "'string'",
])
def test_unsafe_expressions_rejected(bad):
    with pytest.raises(MetricSpecError):
        compile_expr(bad)


def test_valid_expression_compiles():
    compile_expr("abs(plan_margin_ratio - 1) * 2 + (lux > 0.5)")


# ---------------- spec validation (R-8 etc.) ----------------

def test_r8_requires_improvement_action():
    with pytest.raises(MetricSpecError, match="improvement action"):
        validate_spec({"metric_id": "custom.x", "definition": "d", "expr": "lux",
                       "agg": "mean", "improvement_actions": []})


def test_definition_required():
    with pytest.raises(MetricSpecError):
        validate_spec({"metric_id": "custom.x", "definition": "", "expr": "lux",
                       "agg": "mean", "improvement_actions": ["a"]})


def test_bad_agg_rejected():
    with pytest.raises(MetricSpecError):
        validate_spec({"metric_id": "custom.x", "definition": "d", "expr": "lux",
                       "agg": "p99", "improvement_actions": ["a"]})


def test_improvement_actions_accepts_newline_string():
    spec = validate_spec({"metric_id": "custom.x", "definition": "d", "expr": "lux",
                          "agg": "mean", "improvement_actions": "first\n\nsecond"})
    assert spec["improvement_actions"] == ["first", "second"]


# ---------------- persistence ----------------

def test_store_add_list_delete(tmp_path):
    store = CustomMetricStore(tmp_path / "cm.json")
    store.add({"metric_id": "custom.a", "definition": "d", "expr": "lux", "agg": "mean",
               "improvement_actions": ["a"]})
    assert [m["metric_id"] for m in store.list()] == ["custom.a"]
    # persisted across instances
    assert [m["metric_id"] for m in CustomMetricStore(tmp_path / "cm.json").list()] == ["custom.a"]
    assert store.delete("custom.a") is True
    assert store.list() == []
    assert store.delete("custom.a") is False


def test_store_rejects_duplicate(tmp_path):
    store = CustomMetricStore(tmp_path / "cm.json")
    store.add({"metric_id": "custom.a", "definition": "d", "expr": "lux", "agg": "mean",
               "improvement_actions": ["a"]})
    with pytest.raises(MetricSpecError, match="already exists"):
        store.add({"metric_id": "custom.a", "definition": "d2", "expr": "lux", "agg": "mean",
                   "improvement_actions": ["a"]})