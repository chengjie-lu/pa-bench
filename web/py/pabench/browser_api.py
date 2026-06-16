"""In-browser API backend (Pyodide): mirrors the platform API over an in-memory store.

This lets the static GitHub Pages site set up + run evaluations and register custom metrics with
NO server — the whole pabench pipeline runs client-side in WebAssembly. It is pure Python
(numpy + stdlib); the FastAPI platform/ layer is deliberately NOT imported in the browser.

The single entry point is BrowserBackend.handle(method, path, query, body) -> (status, obj),
which the Pyodide worker calls for each intercepted /api/* request. Runs execute synchronously
(FakeSim is ~1s), so a created run is already "done" by the time POST returns — no SSE needed.
"""
from __future__ import annotations

from datetime import date

from .schema import BENCHMARK_VERSION, Episode
from .pipeline import (HW_REGISTRY, MODEL_REGISTRY, RunConfig, build_scenes,
                       run_benchmark)
from .runners import BACKEND_IDS
from .metrics import (AGGS, METRIC_FIELDS, MetricSpecError, compute_for_combos,
                      validate_registry, validate_spec)
from .metrics.registry import METRIC_REGISTRY
from .webdata import build_index, index_record

SORTABLE = {"episode_id", "model_id", "hw_config_id", "lux", "success",
            "duration_s", "plan_margin_ratio", "e_track_steady_rms_mm", "peak_uncertainty"}
LEGACY_RUN_ID = "run-000-legacy-out"


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _calib_quarter(hw_config_id: str):
    tail = hw_config_id.rsplit("-", 1)[-1]
    if len(tail) == 6 and tail[:4].isdigit() and tail[4] == "Q" and tail[5].isdigit():
        y, q = int(tail[:4]), int(tail[5])
        today = date.today()
        age_quarters = (today.year - y) * 4 + ((today.month - 1) // 3 + 1 - q)
        return tail, age_quarters > 4
    return None, False


def _backends_status():
    try:
        from .runners.mujoco_sim import MUJOCO_AVAILABLE
    except Exception:
        MUJOCO_AVAILABLE = False
    try:
        from .runners.agx_sim import AGX_AVAILABLE
    except Exception:
        AGX_AVAILABLE = False
    out = [{"id": "fake", "available": True, "note": "analytic stub, runs in-browser"},
           {"id": "mujoco", "available": bool(MUJOCO_AVAILABLE),
            "note": "MuJoCo" if MUJOCO_AVAILABLE else "not available in the browser"},
           {"id": "agx", "available": bool(AGX_AVAILABLE),
            "note": "AGX Dynamics" if AGX_AVAILABLE else "not available in the browser"}]
    return [b for b in out if b["id"] in BACKEND_IDS]


class _Run:
    def __init__(self, run_id: str, cfg: RunConfig, created_at: str):
        self.run_id = run_id
        self.cfg = cfg
        self.created_at = created_at
        self.status = "running"
        self.done = 0
        self.total = cfg.total_episodes()
        self.combo_progress: dict[str, int] = {}
        self.error = None
        self.report = None        # {meta, results}
        self.index = None         # build_index output
        self.episodes: dict[str, dict] = {}  # episode_id -> full dict

    def meta(self):
        return {"run_id": self.run_id, "status": self.status, "created_at": self.created_at,
                "config": self.cfg.to_dict(), "done_episodes": self.done,
                "total_episodes": self.total,
                "combos": len(self.cfg.model_ids) * len(self.cfg.hw_ids),
                "combo_progress": dict(self.combo_progress), "error": self.error,
                "benchmark_version": BENCHMARK_VERSION}


class BrowserBackend:
    def __init__(self):
        validate_registry()
        self.runs: dict[str, _Run] = {}
        self.custom: list[dict] = []
        self._seq = 0

    # ---- seeding from the static web/data demo + saved custom metrics ----
    def load_legacy(self, report: dict, index: dict, episodes: list[dict] | None = None):
        cfg = RunConfig(seed=report.get("meta", {}).get("seed", 7))
        run = _Run(LEGACY_RUN_ID, cfg, "0000")
        run.report = report
        run.index = index
        run.total = run.done = len(index.get("episodes", []))
        for ep in episodes or []:
            run.episodes[ep["episode_id"]] = ep
        run.status = "done"
        self.runs[LEGACY_RUN_ID] = run

    def seed_custom(self, specs):
        self.custom = []
        for s in specs or []:
            try:
                self.custom.append(validate_spec(s))
            except MetricSpecError:
                pass  # drop anything that no longer validates

    # ---- request dispatch ----
    def handle(self, method: str, path: str, query: dict | None, body: dict | None):
        query = query or {}
        try:
            return self._route(method, path, query, body)
        except ApiError as e:
            return e.status, {"detail": e.detail}
        except MetricSpecError as e:
            return 400, {"detail": str(e)}
        except Exception as e:  # surfaced to the UI like a 500
            return 500, {"detail": f"{type(e).__name__}: {e}"}

    def _route(self, method, path, query, body):
        seg = [s for s in path.split("/") if s]  # ['api', ...]
        assert seg and seg[0] == "api"
        seg = seg[1:]
        m = method.upper()

        if seg == ["ping"]:
            return 200, {"service": "pa-bench-browser", "benchmark_version": BENCHMARK_VERSION,
                         "milestone": "M-FE2", "runtime": "pyodide"}
        if seg == ["models"]:
            return 200, {"models": [{"model_id": mid, "provides_uncertainty": True, "fake": True}
                                    for mid in MODEL_REGISTRY]}
        if seg == ["hardware"]:
            out = []
            for hid, hw in HW_REGISTRY.items():
                quarter, stale = _calib_quarter(hid)
                out.append({"hw_config_id": hid, "calibrated": quarter, "stale": stale,
                            "tracking_alpha": hw.tracking_alpha, "jitter_amp": hw.jitter_amp})
            return 200, {"hardware": out}
        if seg == ["backends"]:
            return 200, {"backends": _backends_status()}
        if seg == ["metric-registry"]:
            return 200, METRIC_REGISTRY
        if seg == ["metric-fields"]:
            return 200, {"fields": [{"name": k, "description": v} for k, v in METRIC_FIELDS.items()],
                         "aggregations": list(AGGS)}

        if seg == ["custom-metrics"]:
            if m == "GET":
                return 200, {"metrics": list(self.custom)}
            if m == "POST":
                clean = validate_spec(body or {})
                if any(s["metric_id"] == clean["metric_id"] for s in self.custom):
                    raise ApiError(400, f"metric_id already exists: {clean['metric_id']}")
                self.custom.append(clean)
                return 201, clean
        if len(seg) == 2 and seg[0] == "custom-metrics" and m == "DELETE":
            mid = seg[1]
            before = len(self.custom)
            self.custom = [s for s in self.custom if s["metric_id"] != mid]
            if len(self.custom) == before:
                raise ApiError(404, f"custom metric not found: {mid}")
            return 200, {"deleted": mid}

        if seg == ["scenes", "preview"] and m == "POST":
            n = min(int((body or {}).get("n", 12)), 50)
            cfg = RunConfig.from_dict(dict(body or {}, nominal=True, mutation_episodes=max(n, 1)))
            scenes, _ = build_scenes(cfg)
            return 200, {"scenes": [{"scene_id": s.scene_id, "perturbation": s.perturbation}
                                    for s in scenes[1:n + 1]]}

        if seg == ["runs"]:
            if m == "GET":
                runs = sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)
                return 200, {"runs": [r.meta() for r in runs]}
            if m == "POST":
                return self._create_run(body or {})

        if len(seg) >= 2 and seg[0] == "runs":
            run = self.runs.get(self._resolve(seg[1]))
            if not run:
                raise ApiError(404, f"run not found: {seg[1]}")
            rest = seg[2:]
            if not rest and m == "GET":
                return 200, run.meta()
            if rest == ["cancel"] and m == "POST":
                raise ApiError(409, f"run is already in a terminal state: {run.status}")
            if rest == ["progress"] and m == "GET":
                return 200, {"run_id": run.run_id, "status": run.status, "done": run.done,
                             "total": run.total, "combo_progress": dict(run.combo_progress),
                             "episodes_per_combo": run.cfg.episodes_per_combo(),
                             "error": run.error, "events": []}
            if rest == ["summary"] and m == "GET":
                if run.report is None:
                    raise ApiError(409, f"run not finished (status={run.status})")
                return 200, run.report
            if rest == ["index"] and m == "GET":
                if run.index is None:
                    raise ApiError(409, f"run not finished (status={run.status})")
                custom = compute_for_combos(self.custom, run.index["episodes"]) if self.custom else {}
                return 200, dict(run.index, run_id=run.run_id,
                                 custom_metrics=custom, custom_metric_specs=list(self.custom))

        if seg == ["episodes"] and m == "GET":
            return self._list_episodes(query)
        if len(seg) == 2 and seg[0] == "episodes" and m == "GET":
            eid = seg[1]
            for run in self.runs.values():
                if eid in run.episodes:
                    return 200, run.episodes[eid]
            raise ApiError(404, f"episode not found: {eid}")

        raise ApiError(404, f"no route: {method} {path}")

    # ---- helpers ----
    def _resolve(self, rid: str) -> str:
        if rid == "latest":
            done = [r for r in sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)
                    if r.status == "done"]
            return done[0].run_id if done else rid
        return rid

    def _create_run(self, body: dict):
        cfg = RunConfig.from_dict(body)
        if cfg.total_episodes() > 5000:
            raise ApiError(400, f"total episodes {cfg.total_episodes()} exceeds the cap of 5000")
        self._seq += 1
        run = _Run(f"run-browser-{self._seq:04d}", cfg, f"{self._seq:04d}")
        self.runs[run.run_id] = run

        def on_event(ev):
            if ev.get("type") == "episode_done":
                run.done = ev["done"]
                run.combo_progress[ev["combo"]] = run.combo_progress.get(ev["combo"], 0) + 1

        result = run_benchmark(cfg, on_event=on_event)
        records = [index_record(ep) for ep in result["episodes"]]
        run.report = {"meta": result["meta"], "results": result["results"]}
        run.index = build_index(run.report, records)
        run.episodes = {ep.episode_id: ep.to_dict() for ep in result["episodes"]}
        run.status = "done"
        return 201, {"run_id": run.run_id, "total_episodes": run.total}

    def _list_episodes(self, query):
        run = self.runs.get(self._resolve(query.get("run", "latest")))
        if not run:
            raise ApiError(404, f"run not found: {query.get('run')}")
        if run.index is None:
            raise ApiError(409, f"run not finished (status={run.status})")
        recs = run.index["episodes"]
        eq = {"model_id": query.get("model_id"), "hw_config_id": query.get("hw_config_id"),
              "task_type": query.get("task_type"), "tolerance_class": query.get("tolerance_class"),
              "generation_method": query.get("generation_method"),
              "failure_phase": query.get("failure_phase"), "attribution": query.get("attribution"),
              "parent_episode_id": query.get("parent_episode_id")}
        for k, v in eq.items():
            if v is not None and v != "":
                recs = [x for x in recs if x.get(k) == v]
        if query.get("success") not in (None, ""):
            want = str(query["success"]).lower() == "true"
            recs = [x for x in recs if x["success"] == want]
        if query.get("lux_min") not in (None, ""):
            recs = [x for x in recs if x["lux"] >= float(query["lux_min"])]
        if query.get("lux_max") not in (None, ""):
            recs = [x for x in recs if x["lux"] <= float(query["lux_max"])]
        sort = query.get("sort", "episode_id")
        if sort not in SORTABLE:
            raise ApiError(400, f"unsupported sort column: {sort}")
        recs = sorted(recs, key=lambda x: (x[sort] is None, x[sort]),
                      reverse=(query.get("order") == "desc"))
        page = int(query.get("page", 1)); size = int(query.get("size", 50))
        start = (page - 1) * size
        return 200, {"run_id": run.run_id, "total": len(recs), "page": page, "size": size,
                     "episodes": recs[start:start + size]}