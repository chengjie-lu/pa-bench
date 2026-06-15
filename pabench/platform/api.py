"""FastAPI 包装 (fe-rq.md §8 契约): pabench 流水线的发起 / 进度 SSE / 结果消费。

M-FE2 范围: 不含场景编辑器与 oracle-replay (M-FE3)。
所有写操作预留 auth header 位 (NFR-FE N7) —— V1 内网免登录, 不校验。
静态前端 (web/) 由同一进程挂载, 浏览器同源直连 /api/*。
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ..schema import BENCHMARK_VERSION
from ..pipeline import HW_REGISTRY, MODEL_REGISTRY, RunConfig, build_scenes
from ..metrics import validate_registry
from ..metrics.registry import METRIC_REGISTRY
from .run_manager import RunManager, TERMINAL

ROOT = Path(__file__).resolve().parent.parent.parent  # pa-bench/

# 回合列表可排序列 (服务端排序, fe-rq.md §6: 数据可能 >10k 回合)
SORTABLE = {"episode_id", "model_id", "hw_config_id", "lux", "success",
            "duration_s", "plan_margin_ratio", "e_track_steady_rms_mm",
            "peak_uncertainty"}
# 大数组字段 (stride 降采样目标, NFR-FE N2)
_MODEL_ARRAYS = ["t", "cmd_xyz", "cmd_yaw", "entropy", "latency_ms"]
_ROBOT_ARRAYS = ["t", "ee_xyz_actual", "ee_yaw_actual", "ft_wrench", "gripper_width"]


def _stride_episode(d: dict, stride: int) -> dict:
    """按 stride 降采样回合大数组 (§8 调试页先 stride=4 快渲染再拉全量)。"""
    if stride <= 1:
        return d
    d = json.loads(json.dumps(d))  # 深拷贝, 不污染磁盘缓存
    for k in _MODEL_ARRAYS:
        v = d["model"]["actions"].get(k)
        if v is not None:
            d["model"]["actions"][k] = v[::stride]
    for k in _ROBOT_ARRAYS:
        v = d["robot"].get(k)
        if v is not None:
            d["robot"][k] = v[::stride]
    d["robot"]["phase_spans"] = [[p, i0 // stride, i1 // stride]
                                 for p, i0, i1 in d["robot"]["phase_spans"]]
    d["_stride"] = stride
    return d


def _calib_quarter(hw_config_id: str) -> tuple[str | None, bool]:
    """从 hw_config_id 尾缀解析标定季度 (如 2023Q1); 距今 >4 个季度 → 过期标黄 (§4.2)。"""
    tail = hw_config_id.rsplit("-", 1)[-1]
    if len(tail) == 6 and tail[:4].isdigit() and tail[4] == "Q" and tail[5].isdigit():
        y, q = int(tail[:4]), int(tail[5])
        today = date.today()
        age_quarters = (today.year - y) * 4 + ((today.month - 1) // 3 + 1 - q)
        return tail, age_quarters > 4
    return None, False


def create_app(runs_dir: Path | None = None, web_dir: Path | None = None,
               legacy_out: Path | None = None) -> FastAPI:
    validate_registry()  # R-8: 服务起不来好过带病上线
    manager = RunManager(runs_dir or ROOT / "runs",
                         legacy_out if legacy_out is not None else ROOT / "out")
    app = FastAPI(title="PA-Bench Platform API", version=BENCHMARK_VERSION,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.manager = manager

    def _run_or_404(run_id: str):
        run = manager.get(run_id)
        if not run:
            raise HTTPException(404, f"运行不存在: {run_id}")
        return run

    # ---------------- 元信息 ----------------
    @app.get("/api/ping")
    def ping():
        """前端模式探测: 200 → server 模式, 失败 → M-FE1 静态模式。"""
        return {"service": "pa-bench-platform", "benchmark_version": BENCHMARK_VERSION,
                "milestone": "M-FE2"}

    @app.get("/api/models")
    def models():
        """向导步骤① 模型多选 (FR-2.4: 注册即被发现)。"""
        return {"models": [
            {"model_id": mid,
             # 两个假模型都输出 entropy; 真模型缺失时 ActionChunk.entropy=None → False
             "provides_uncertainty": True,
             "fake": True}
            for mid in MODEL_REGISTRY]}

    @app.get("/api/hardware")
    def hardware():
        """向导步骤① 硬件档案多选 (标定过期标黄, NFR-2)。"""
        out = []
        for hid, hw in HW_REGISTRY.items():
            quarter, stale = _calib_quarter(hid)
            out.append({"hw_config_id": hid, "calibrated": quarter, "stale": stale,
                        "tracking_alpha": hw.tracking_alpha,
                        "jitter_amp": hw.jitter_amp})
        return {"hardware": out}

    @app.get("/api/metric-registry")
    def metric_registry():
        return METRIC_REGISTRY

    @app.post("/api/scenes/preview")
    async def scenes_preview(request: Request):
        """向导步骤②「预览采样」: 返回前 N 个采样场景的参数卡, 不执行 (§4.2)。"""
        body = await request.json()
        n = min(int(body.get("n", 12)), 50)
        try:
            cfg = RunConfig.from_dict(dict(body, nominal=True,
                                           mutation_episodes=max(n, 1)))
        except ValueError as e:
            raise HTTPException(400, str(e))
        scenes, _ = build_scenes(cfg)
        return {"scenes": [
            {"scene_id": s.scene_id, "perturbation": s.perturbation}
            for s in scenes[1:n + 1]]}

    # ---------------- 运行 ----------------
    @app.get("/api/runs")
    def list_runs():
        return {"runs": [dict(r.meta(), benchmark_version=BENCHMARK_VERSION)
                         for r in manager.list_runs()]}

    @app.post("/api/runs", status_code=201)
    async def create_run(request: Request):
        """启动体 = 向导全部选项 + seed (FR-FE-2.1)。返回 run_id, 执行在后台线程。"""
        body = await request.json()
        try:
            cfg = RunConfig.from_dict(body)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        if cfg.total_episodes() > 5000:
            raise HTTPException(400, f"总回合数 {cfg.total_episodes()} 超出上限 5000")
        run = manager.start_run(cfg)
        return {"run_id": run.run_id, "total_episodes": run.total}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        return dict(_run_or_404(run_id).meta(), benchmark_version=BENCHMARK_VERSION)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        run = _run_or_404(run_id)
        if not manager.cancel(run.run_id):
            raise HTTPException(409, f"运行已是终态: {run.status}")
        return {"run_id": run.run_id, "cancelling": True}

    @app.get("/api/runs/{run_id}/progress")
    def progress(run_id: str):
        """SSE 断线降级轮询端点 (§8): 状态 + 计数 + 最近 20 条事件。"""
        run = _run_or_404(run_id)
        return {"run_id": run.run_id, "status": run.status,
                "done": run.done, "total": run.total,
                "combo_progress": dict(run.combo_progress),
                "episodes_per_combo": run.cfg.episodes_per_combo(),
                "error": run.error, "events": run.events_since(0)[-20:]}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str):
        """SSE: 先重放事件历史再实时推送; 终态后发 run_closed 收尾 (§11)。"""
        run = _run_or_404(run_id)

        async def stream():
            seq = 0
            while True:
                batch = run.events_since(seq)
                for e in batch:
                    yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                    seq = e["seq"] + 1
                if run.status in TERMINAL and not run.events_since(seq):
                    yield (f"event: run_closed\ndata: "
                           f"{json.dumps({'status': run.status}, ensure_ascii=False)}\n\n")
                    return
                # 阻塞等待新事件 (线程信号), 2s 超时发心跳注释保活
                got = await run_in_threadpool(run.wait_new, seq, 2.0)
                if not got:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0)  # 让出事件循环

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}/summary")
    def summary(run_id: str):
        """= report.json (meta + results[]) (§8)。"""
        run = _run_or_404(run_id)
        path = run.dir / "report.json"
        if not path.exists():
            raise HTTPException(409, f"运行未完成 (status={run.status}), 暂无 summary")
        return json.loads(path.read_text())

    @app.get("/api/runs/{run_id}/index")
    def run_index(run_id: str):
        """回合索引 + C1/C2/C3/C5 预聚合 — 与 M-FE1 web/data/index.json 同构,
        前端适配层零改动复用现有视图 (聚合在后端做, §8)。"""
        run = _run_or_404(run_id)
        index = manager.load_index(run)
        if index is None:
            raise HTTPException(409, f"运行未完成 (status={run.status}), 暂无 index")
        return dict(index, run_id=run.run_id)

    # ---------------- 回合 ----------------
    @app.get("/api/episodes")
    def list_episodes(
        run: str = Query("latest"), model_id: str | None = None,
        hw_config_id: str | None = None, task_type: str | None = None,
        tolerance_class: str | None = None, generation_method: str | None = None,
        success: bool | None = None, failure_phase: str | None = None,
        attribution: str | None = None, parent_episode_id: str | None = None,
        lux_min: float | None = None, lux_max: float | None = None,
        sort: str = "episode_id", order: str = "asc",
        page: int = Query(1, ge=1), size: int = Query(50, ge=1, le=500),
    ):
        """服务端筛选 + 排序 + 分页 (§4.4 全部筛选项; 列表永不返回大数组, N2)。"""
        r = _run_or_404(run)
        index = manager.load_index(r)
        if index is None:
            raise HTTPException(409, f"运行未完成 (status={r.status})")
        recs = index["episodes"]
        eq_filters = {"model_id": model_id, "hw_config_id": hw_config_id,
                      "task_type": task_type, "tolerance_class": tolerance_class,
                      "generation_method": generation_method,
                      "failure_phase": failure_phase, "attribution": attribution,
                      "parent_episode_id": parent_episode_id}
        for key, val in eq_filters.items():
            if val is not None:
                recs = [x for x in recs if x[key] == val]
        if success is not None:
            recs = [x for x in recs if x["success"] == success]
        if lux_min is not None:
            recs = [x for x in recs if x["lux"] >= lux_min]
        if lux_max is not None:
            recs = [x for x in recs if x["lux"] <= lux_max]
        if sort not in SORTABLE:
            raise HTTPException(400, f"不支持的排序列: {sort}")
        recs = sorted(recs, key=lambda x: (x[sort] is None, x[sort]),
                      reverse=(order == "desc"))
        total = len(recs)
        start = (page - 1) * size
        return {"run_id": r.run_id, "total": total, "page": page, "size": size,
                "episodes": recs[start:start + size]}

    @app.get("/api/episodes/{episode_id}")
    def episode_detail(episode_id: str, stride: int = Query(1, ge=1, le=64)):
        """单回合全量 (含大数组), stride 降采样走独立请求 (§8/N2)。"""
        path = manager.find_episode(episode_id)
        if path is None:
            raise HTTPException(404, f"回合不存在: {episode_id}")
        return _stride_episode(json.loads(path.read_text()), stride)

    # ---------------- 静态前端 (放最后, /api 优先匹配) ----------------
    web = web_dir if web_dir is not None else ROOT / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
