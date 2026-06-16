"""M-FE2 platform API end-to-end (fe-rq.md §8/§9): launch run → progress → result consumption.

Covers FR-FE-2.1 (seed pass-through reproducibility), episode filter/sort/pagination, stride downsampling, cancel,
scenes/preview, and the models/hardware lists. FakeSim is fast, so everything waits synchronously inside TestClient.
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pabench.platform import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    # legacy_out=None: do not import out/, each test starts from an empty run set
    app = create_app(runs_dir=tmp_path / "runs", web_dir=None, legacy_out=None)
    return TestClient(app)


def _wait_done(client, run_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = client.get(f"/api/runs/{run_id}/progress").json()
        if p["status"] in ("done", "failed", "cancelled"):
            return p
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} timed out before completing")


def _start(client, **cfg):
    cfg.setdefault("mutation_episodes", 4)
    r = client.post("/api/runs", json=cfg)
    assert r.status_code == 201, r.text
    return r.json()["run_id"]


# ---------------- metadata endpoints ----------------
def test_ping(client):
    assert client.get("/api/ping").json()["milestone"] == "M-FE2"


def test_models_and_hardware(client):
    models = client.get("/api/models").json()["models"]
    assert {m["model_id"] for m in models} == {"precise-vla-0.3", "sloppy-vla-0.1"}
    hw = client.get("/api/hardware").json()["hardware"]
    ids = {h["hw_config_id"]: h for h in hw}
    assert "arm-calibrated-2026Q2" in ids and "arm-worn-2023Q1" in ids
    # 2023Q1 is stale by 2026 (flagged yellow), 2026Q2 is not stale (NFR-2)
    assert ids["arm-worn-2023Q1"]["stale"] is True
    assert ids["arm-calibrated-2026Q2"]["stale"] is False


def test_metric_registry(client):
    reg = client.get("/api/metric-registry").json()
    assert reg and all(v.get("improvement_actions") for v in reg.values())


def test_scenes_preview(client):
    body = {"seed": 7, "n": 5, "pos_range_m": 0.02, "lux_range": [0.4, 0.9]}
    scenes = client.post("/api/scenes/preview", json=body).json()["scenes"]
    assert len(scenes) == 5
    p = scenes[0]["perturbation"]
    assert {"part_dx", "part_dy", "part_dyaw", "lux_factor", "friction"} <= p.keys()
    assert 0.4 <= p["lux_factor"] <= 0.9  # slider range passed through (§4.2)


# ---------------- run lifecycle ----------------
def test_run_lifecycle_and_summary(client):
    rid = _start(client, model_ids=["precise-vla-0.3"], hw_ids=["arm-worn-2023Q1"],
                 seed=3, mutation_episodes=6)
    p = _wait_done(client, rid)
    assert p["status"] == "done"
    assert p["done"] == p["total"] == 1 + 6 + 3  # nominal + mutation + MR

    summary = client.get(f"/api/runs/{rid}/summary").json()
    assert len(summary["results"]) == 1
    r = summary["results"][0]
    assert 0.0 <= r["sr"] <= 1.0
    assert r["ci95"][0] <= r["sr"] <= r["ci95"][1]
    assert summary["meta"]["seed"] == 3

    # index = episode index + pre-aggregations; list rows carry no large arrays (NFR-FE N2)
    index = client.get(f"/api/runs/{rid}/index").json()
    assert len(index["episodes"]) == p["total"]
    assert set(index["aggregates"]) == {"radar", "robustness", "sankey", "failure_hist"}
    assert "ee_xyz_actual" not in index["episodes"][0]


def test_run_appears_in_list(client):
    rid = _start(client, seed=1)
    _wait_done(client, rid)
    runs = client.get("/api/runs").json()["runs"]
    assert any(r["run_id"] == rid and r["status"] == "done" for r in runs)


def test_seed_reproducible(client):
    """FR-FE-2.1: two runs with a fixed seed → identical report (NFR-1 pass-through)."""
    cfg = dict(seed=77, mutation_episodes=5)
    rid1 = _start(client, **cfg); _wait_done(client, rid1)
    rid2 = _start(client, **cfg); _wait_done(client, rid2)
    s1 = client.get(f"/api/runs/{rid1}/summary").json()["results"]
    s2 = client.get(f"/api/runs/{rid2}/summary").json()["results"]
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)


def test_different_seed_differs(client):
    rid1 = _start(client, seed=1, mutation_episodes=5); _wait_done(client, rid1)
    rid2 = _start(client, seed=2, mutation_episodes=5); _wait_done(client, rid2)
    s1 = client.get(f"/api/runs/{rid1}/summary").json()["results"]
    s2 = client.get(f"/api/runs/{rid2}/summary").json()["results"]
    assert json.dumps(s1, sort_keys=True) != json.dumps(s2, sort_keys=True)


# ---------------- episode list: filter / sort / pagination ----------------
def test_episodes_filter_sort_page(client):
    rid = _start(client, seed=7, mutation_episodes=12)
    _wait_done(client, rid)

    all_eps = client.get("/api/episodes", params={"run": rid, "size": 500}).json()
    total = all_eps["total"]
    assert total == 4 * (1 + 12 + 3)  # default 2 models × 2 hardware = 4 combos

    # success filter: a subset, all successful
    succ = client.get("/api/episodes", params={"run": rid, "success": "true", "size": 500}).json()
    assert all(e["success"] for e in succ["episodes"])
    fail = client.get("/api/episodes", params={"run": rid, "success": "false", "size": 500}).json()
    assert succ["total"] + fail["total"] == total

    # model filter
    m = client.get("/api/episodes",
                   params={"run": rid, "model_id": "sloppy-vla-0.1", "size": 500}).json()
    assert all(e["model_id"] == "sloppy-vla-0.1" for e in m["episodes"])

    # sort: e_track descending
    s = client.get("/api/episodes", params={"run": rid, "sort": "e_track_steady_rms_mm",
                                            "order": "desc", "size": 500}).json()
    vals = [e["e_track_steady_rms_mm"] for e in s["episodes"]]
    assert vals == sorted(vals, reverse=True)

    # pagination: two pages reassemble the full set
    p1 = client.get("/api/episodes", params={"run": rid, "page": 1, "size": 10}).json()
    p2 = client.get("/api/episodes", params={"run": rid, "page": 2, "size": 10}).json()
    assert len(p1["episodes"]) == 10 and p1["total"] == total
    ids = {e["episode_id"] for e in p1["episodes"]} | {e["episode_id"] for e in p2["episodes"]}
    assert len(ids) == min(20, total)


def test_episodes_bad_sort_400(client):
    rid = _start(client, seed=7)
    _wait_done(client, rid)
    r = client.get("/api/episodes", params={"run": rid, "sort": "; DROP"})
    assert r.status_code == 400


def test_episode_detail_stride(client):
    rid = _start(client, seed=7)
    _wait_done(client, rid)
    eid = client.get("/api/episodes", params={"run": rid}).json()["episodes"][0]["episode_id"]
    full = client.get(f"/api/episodes/{eid}").json()
    n = len(full["robot"]["t"])
    ds = client.get(f"/api/episodes/{eid}", params={"stride": 4}).json()
    assert len(ds["robot"]["t"]) == (n + 3) // 4
    assert len(ds["model"]["actions"]["cmd_xyz"]) == (n + 3) // 4
    assert ds["_stride"] == 4
    # phase_spans indices scale in sync
    assert ds["robot"]["phase_spans"][0][2] <= len(ds["robot"]["t"])


def test_episode_404(client):
    assert client.get("/api/episodes/does-not-exist").status_code == 404


# ---------------- cancel ----------------
def test_cancel_run(client):
    # large budget + throttle → enough time to send a cancel
    rid = _start(client, seed=5, mutation_episodes=60, pace_s=0.03)
    time.sleep(0.3)
    assert client.post(f"/api/runs/{rid}/cancel").json()["cancelling"] is True
    p = _wait_done(client, rid)
    assert p["status"] == "cancelled"
    assert p["done"] < p["total"]
    # a cancelled run has no summary (artifacts not persisted)
    assert client.get(f"/api/runs/{rid}/summary").status_code == 409


# ---------------- validation errors ----------------
def test_invalid_config_400(client):
    # metamorphic requires nominal
    r = client.post("/api/runs", json={"nominal": False, "mutation_episodes": 4,
                                       "metamorphic": True})
    assert r.status_code == 400
    # unregistered model
    r = client.post("/api/runs", json={"model_ids": ["ghost-vla"]})
    assert r.status_code == 400


def test_missing_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/summary").status_code == 404


# ---------------- historical-artifact import ----------------
def test_legacy_import(tmp_path):
    """out/ artifacts are registered as a completed run (M-FE1 → M-FE2 transition)."""
    out = Path(__file__).resolve().parent.parent / "out"
    if not (out / "episodes.jsonl").exists():
        pytest.skip("no out/ artifacts (run demo.py first)")
    app = create_app(runs_dir=tmp_path / "runs", web_dir=None, legacy_out=out)
    c = TestClient(app)
    runs = c.get("/api/runs").json()["runs"]
    legacy = [r for r in runs if r["run_id"] == "run-000-legacy-out"]
    assert legacy and legacy[0]["status"] == "done"
    assert c.get("/api/runs/run-000-legacy-out/summary").status_code == 200
