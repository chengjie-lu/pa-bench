"""M-FE2 platform API 端到端 (fe-rq.md §8/§9): 发起运行 → 进度 → 结果消费。

覆盖 FR-FE-2.1 (seed 透传可复现)、回合筛选/排序/分页、stride 降采样、取消、
scenes/preview、models/hardware 列表。FakeSim 快, 全程在 TestClient 内同步等待。
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
    # legacy_out=None: 不导入 out/, 每个用例从空运行集起步
    app = create_app(runs_dir=tmp_path / "runs", web_dir=None, legacy_out=None)
    return TestClient(app)


def _wait_done(client, run_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = client.get(f"/api/runs/{run_id}/progress").json()
        if p["status"] in ("done", "failed", "cancelled"):
            return p
        time.sleep(0.02)
    raise AssertionError(f"运行 {run_id} 超时未完成")


def _start(client, **cfg):
    cfg.setdefault("mutation_episodes", 4)
    r = client.post("/api/runs", json=cfg)
    assert r.status_code == 201, r.text
    return r.json()["run_id"]


# ---------------- 元信息端点 ----------------
def test_ping(client):
    assert client.get("/api/ping").json()["milestone"] == "M-FE2"


def test_models_and_hardware(client):
    models = client.get("/api/models").json()["models"]
    assert {m["model_id"] for m in models} == {"precise-vla-0.3", "sloppy-vla-0.1"}
    hw = client.get("/api/hardware").json()["hardware"]
    ids = {h["hw_config_id"]: h for h in hw}
    assert "arm-calibrated-2026Q2" in ids and "arm-worn-2023Q1" in ids
    # 2023Q1 距 2026 已过期标黄, 2026Q2 不过期 (NFR-2)
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
    assert 0.4 <= p["lux_factor"] <= 0.9  # 滑杆区间透传 (§4.2)


# ---------------- 运行生命周期 ----------------
def test_run_lifecycle_and_summary(client):
    rid = _start(client, model_ids=["precise-vla-0.3"], hw_ids=["arm-worn-2023Q1"],
                 seed=3, mutation_episodes=6)
    p = _wait_done(client, rid)
    assert p["status"] == "done"
    assert p["done"] == p["total"] == 1 + 6 + 3  # nominal + mut + MR

    summary = client.get(f"/api/runs/{rid}/summary").json()
    assert len(summary["results"]) == 1
    r = summary["results"][0]
    assert 0.0 <= r["sr"] <= 1.0
    assert r["ci95"][0] <= r["sr"] <= r["ci95"][1]
    assert summary["meta"]["seed"] == 3

    # index = 回合索引 + 预聚合, 列表行不含大数组 (NFR-FE N2)
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
    """FR-FE-2.1: 固定 seed 两次运行 → report 完全一致 (NFR-1 透传)。"""
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


# ---------------- 回合列表: 筛选 / 排序 / 分页 ----------------
def test_episodes_filter_sort_page(client):
    rid = _start(client, seed=7, mutation_episodes=12)
    _wait_done(client, rid)

    all_eps = client.get("/api/episodes", params={"run": rid, "size": 500}).json()
    total = all_eps["total"]
    assert total == 4 * (1 + 12 + 3)  # 默认 2 模型 × 2 硬件 = 4 组合

    # 成功筛选: 子集且全 success
    succ = client.get("/api/episodes", params={"run": rid, "success": "true", "size": 500}).json()
    assert all(e["success"] for e in succ["episodes"])
    fail = client.get("/api/episodes", params={"run": rid, "success": "false", "size": 500}).json()
    assert succ["total"] + fail["total"] == total

    # 模型筛选
    m = client.get("/api/episodes",
                   params={"run": rid, "model_id": "sloppy-vla-0.1", "size": 500}).json()
    assert all(e["model_id"] == "sloppy-vla-0.1" for e in m["episodes"])

    # 排序: e_track 降序
    s = client.get("/api/episodes", params={"run": rid, "sort": "e_track_steady_rms_mm",
                                            "order": "desc", "size": 500}).json()
    vals = [e["e_track_steady_rms_mm"] for e in s["episodes"]]
    assert vals == sorted(vals, reverse=True)

    # 分页: 两页拼回全集
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
    # phase_spans 索引同步缩放
    assert ds["robot"]["phase_spans"][0][2] <= len(ds["robot"]["t"])


def test_episode_404(client):
    assert client.get("/api/episodes/does-not-exist").status_code == 404


# ---------------- 取消 ----------------
def test_cancel_run(client):
    # 大预算 + 节流 → 有时间发取消
    rid = _start(client, seed=5, mutation_episodes=60, pace_s=0.03)
    time.sleep(0.3)
    assert client.post(f"/api/runs/{rid}/cancel").json()["cancelling"] is True
    p = _wait_done(client, rid)
    assert p["status"] == "cancelled"
    assert p["done"] < p["total"]
    # 取消的运行无 summary (产物未落盘)
    assert client.get(f"/api/runs/{rid}/summary").status_code == 409


# ---------------- 校验错误 ----------------
def test_invalid_config_400(client):
    # metamorphic 需要 nominal
    r = client.post("/api/runs", json={"nominal": False, "mutation_episodes": 4,
                                       "metamorphic": True})
    assert r.status_code == 400
    # 未注册模型
    r = client.post("/api/runs", json={"model_ids": ["ghost-vla"]})
    assert r.status_code == 400


def test_missing_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/summary").status_code == 404


# ---------------- 历史产物导入 ----------------
def test_legacy_import(tmp_path):
    """out/ 产物被注册为已完成运行 (M-FE1 → M-FE2 过渡)。"""
    out = Path(__file__).resolve().parent.parent / "out"
    if not (out / "episodes.jsonl").exists():
        pytest.skip("无 out/ 产物 (先跑 demo.py)")
    app = create_app(runs_dir=tmp_path / "runs", web_dir=None, legacy_out=out)
    c = TestClient(app)
    runs = c.get("/api/runs").json()["runs"]
    legacy = [r for r in runs if r["run_id"] == "run-000-legacy-out"]
    assert legacy and legacy[0]["status"] == "done"
    assert c.get("/api/runs/run-000-legacy-out/summary").status_code == 200
