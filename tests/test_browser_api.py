"""In-browser backend (Pyodide path): the in-memory dispatcher mirrors the platform API.

These run in plain CPython (the dispatcher is pure python); they don't exercise Pyodide itself,
but they lock down the request routing the worker depends on.
"""
import pytest

from pabench.browser_api import BrowserBackend, LEGACY_RUN_ID


@pytest.fixture()
def be():
    return BrowserBackend()


def _ok(resp):
    status, body = resp
    assert status in (200, 201), body
    return body


# ---------------- metadata endpoints ----------------
def test_ping_reports_pyodide(be):
    body = _ok(be.handle("GET", "/api/ping", {}, None))
    assert body["runtime"] == "pyodide" and body["milestone"] == "M-FE2"


def test_models_hardware_backends(be):
    models = _ok(be.handle("GET", "/api/models", {}, None))["models"]
    assert {m["model_id"] for m in models} == {"precise-vla-0.3", "sloppy-vla-0.1"}
    hw = {h["hw_config_id"]: h for h in _ok(be.handle("GET", "/api/hardware", {}, None))["hardware"]}
    assert hw["arm-worn-2023Q1"]["stale"] is True
    backends = {b["id"]: b for b in _ok(be.handle("GET", "/api/backends", {}, None))["backends"]}
    assert backends["fake"]["available"] is True  # fake runs in-browser
    assert backends["mujoco"]["available"] is False  # physics libs unavailable in the browser


def test_metric_fields(be):
    body = _ok(be.handle("GET", "/api/metric-fields", {}, None))
    assert "rate" in body["aggregations"]
    assert {f["name"] for f in body["fields"]} >= {"success", "plan_margin_ratio", "lux"}


# ---------------- run lifecycle (synchronous in-browser) ----------------
def test_run_completes_synchronously_and_serves_data(be):
    status, res = be.handle("POST", "/api/runs", {}, {
        "model_ids": ["precise-vla-0.3"], "hw_ids": ["arm-worn-2023Q1"],
        "seed": 3, "mutation_episodes": 6})
    assert status == 201
    rid = res["run_id"]
    assert res["total_episodes"] == 1 + 6 + 3
    # already done by the time POST returns → no SSE needed
    assert _ok(be.handle("GET", f"/api/runs/{rid}", {}, None))["status"] == "done"
    summary = _ok(be.handle("GET", f"/api/runs/{rid}/summary", {}, None))
    assert len(summary["results"]) == 1 and summary["meta"]["seed"] == 3
    index = _ok(be.handle("GET", f"/api/runs/{rid}/index", {}, None))
    assert len(index["episodes"]) == res["total_episodes"]
    assert set(index["aggregates"]) == {"radar", "robustness", "sankey", "failure_hist"}
    # run shows up in the list and as 'latest'
    assert any(r["run_id"] == rid for r in _ok(be.handle("GET", "/api/runs", {}, None))["runs"])


def test_episode_list_and_detail(be):
    _, res = be.handle("POST", "/api/runs", {}, {"seed": 7, "mutation_episodes": 4})
    rid = res["run_id"]
    listed = _ok(be.handle("GET", "/api/episodes", {"run": rid, "size": 500}, None))
    assert listed["total"] == 4 * (1 + 4 + 3)  # 2 models × 2 hw
    eid = listed["episodes"][0]["episode_id"]
    detail = _ok(be.handle("GET", f"/api/episodes/{eid}", {}, None))
    assert detail["episode_id"] == eid and "robot" in detail
    # filters / bad sort
    succ = _ok(be.handle("GET", "/api/episodes", {"run": rid, "success": "true", "size": 500}, None))
    assert all(e["success"] for e in succ["episodes"])
    assert be.handle("GET", "/api/episodes", {"run": rid, "sort": "; DROP"}, None)[0] == 400


def test_run_cap(be):
    status, body = be.handle("POST", "/api/runs", {}, {"mutation_episodes": 99999})
    assert status == 400 and "exceeds" in body["detail"]


# ---------------- custom metrics ----------------
def test_custom_metric_crud_and_run_index(be):
    spec = {"metric_id": "custom.fail_rate", "level": "L0", "owner": "both",
            "definition": "failure rate", "expr": "1 - success", "agg": "mean",
            "improvement_actions": ["drill down"]}
    assert be.handle("POST", "/api/custom-metrics", {}, spec)[0] == 201
    assert be.handle("POST", "/api/custom-metrics", {}, spec)[0] == 400  # duplicate
    assert be.handle("POST", "/api/custom-metrics", {}, {  # unsafe formula
        "metric_id": "custom.x", "definition": "d", "expr": "__import__('os')",
        "agg": "mean", "improvement_actions": ["a"]})[0] == 400
    _, res = be.handle("POST", "/api/runs", {}, {
        "model_ids": ["sloppy-vla-0.1"], "hw_ids": ["arm-calibrated-2026Q2"],
        "seed": 5, "mutation_episodes": 6})
    index = _ok(be.handle("GET", f"/api/runs/{res['run_id']}/index", {}, None))
    combo = "sloppy-vla-0.1 @ arm-calibrated-2026Q2"
    assert 0.0 <= index["custom_metrics"][combo]["custom.fail_rate"] <= 1.0
    # delete
    assert be.handle("DELETE", "/api/custom-metrics/custom.fail_rate", {}, None)[0] == 200
    assert be.handle("DELETE", "/api/custom-metrics/custom.fail_rate", {}, None)[0] == 404


def test_seed_custom_drops_invalid(be):
    be.seed_custom([
        {"metric_id": "custom.ok", "definition": "d", "expr": "lux", "agg": "mean",
         "improvement_actions": ["a"]},
        {"metric_id": "custom.broken", "definition": "d", "expr": "nope(", "agg": "mean",
         "improvement_actions": ["a"]},  # invalid → dropped
    ])
    ids = [m["metric_id"] for m in _ok(be.handle("GET", "/api/custom-metrics", {}, None))["metrics"]]
    assert ids == ["custom.ok"]


# ---------------- legacy (static demo) run import ----------------
def test_load_legacy_run(be):
    report = {"meta": {"seed": 7, "benchmark_version": "x", "total_episodes": 1,
                       "attr_rules_version": "attr-rules-0.1"}, "results": []}
    index = {"meta": report["meta"], "episodes": [], "aggregates": {}}
    be.load_legacy(report, index)
    assert _ok(be.handle("GET", f"/api/runs/{LEGACY_RUN_ID}", {}, None))["status"] == "done"
    # 'latest' resolves to the completed legacy run
    assert _ok(be.handle("GET", "/api/runs/latest/summary", {}, None))["meta"]["seed"] == 7


def test_unknown_route_404(be):
    assert be.handle("GET", "/api/nope", {}, None)[0] == 404