"""运行管理器: 评测运行的发起 / 进度事件 / 取消 / 落盘 / 历史加载 (fe-rq.md §4.3, §11)。

每个运行一个目录 runs/<run_id>/:
  run.json          运行元数据 (config + status + 进度), 状态变更即写
  report.json       运行级聚合 (完成后)
  index.json        回合索引 + 图表预聚合 (完成后, 结构同 web/data/index.json)
  episodes.jsonl    原始产物
  episodes/<id>.json 单回合全量 (调试页按需加载, NFR-FE N2)

执行在后台线程 (FakeSim 为 CPU 顺序仿真, 单 worker 足够 M1 纵切);
事件历史保存在内存 + SSE 等待方用 Condition 唤醒 (FR-FE-3.1 ≤2s 反映进度)。
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ..pipeline import RunCancelled, RunConfig, run_benchmark
from ..webdata import export_run_data

TERMINAL = {"done", "failed", "cancelled"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Run:
    """单个评测运行的内存态 (事件历史 + 进度计数 + 取消句柄)。"""

    def __init__(self, run_id: str, cfg: RunConfig, run_dir: Path,
                 status: str = "running", created_at: str | None = None):
        self.run_id = run_id
        self.cfg = cfg
        self.dir = run_dir
        self.status = status
        self.created_at = created_at or _now_iso()
        self.done = 0
        self.total = cfg.total_episodes()
        self.combo_progress: dict[str, int] = {}
        self.error: str | None = None
        self.events: list[dict] = []
        self.cancel_event = threading.Event()
        self.cond = threading.Condition()
        self.thread: threading.Thread | None = None

    # ---- 事件流 (SSE 数据源) ----
    def emit(self, event: dict):
        with self.cond:
            event = dict(event, seq=len(self.events), ts=round(time.time(), 3))
            self.events.append(event)
            if event["type"] == "episode_done":
                self.done = event["done"]
                self.combo_progress[event["combo"]] = \
                    self.combo_progress.get(event["combo"], 0) + 1
            elif event["type"] == "combo_start":
                self.combo_progress.setdefault(event["combo"], 0)
            self.cond.notify_all()

    def events_since(self, seq: int) -> list[dict]:
        with self.cond:
            return self.events[seq:]

    def wait_new(self, seq: int, timeout: float) -> bool:
        """阻塞至有新事件或进入终态; 返回是否有新事件 (SSE 心跳判定用)。"""
        with self.cond:
            if len(self.events) > seq:
                return True
            if self.status in TERMINAL:
                return False
            self.cond.wait(timeout)
            return len(self.events) > seq

    def set_status(self, status: str, error: str | None = None):
        with self.cond:
            self.status = status
            self.error = error
            self.cond.notify_all()
        self.save_meta()

    # ---- 持久化 ----
    def meta(self) -> dict:
        return {
            "run_id": self.run_id, "status": self.status,
            "created_at": self.created_at, "config": self.cfg.to_dict(),
            "done_episodes": self.done, "total_episodes": self.total,
            "combos": len(self.cfg.model_ids) * len(self.cfg.hw_ids),
            "combo_progress": dict(self.combo_progress),
            "error": self.error,
        }

    def save_meta(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run.json").write_text(
            json.dumps(self.meta(), ensure_ascii=False, indent=1))

    @staticmethod
    def load(run_dir: Path) -> "Run | None":
        try:
            m = json.loads((run_dir / "run.json").read_text())
            cfg = RunConfig.from_dict(m["config"])
        except (OSError, ValueError, KeyError):
            return None
        run = Run(m["run_id"], cfg, run_dir,
                  status=m["status"], created_at=m["created_at"])
        run.done = m.get("done_episodes", 0)
        run.total = m.get("total_episodes", run.total)
        run.combo_progress = m.get("combo_progress", {})
        run.error = m.get("error")
        if run.status == "running":  # 服务重启时孤儿运行 → failed (线程已不存在)
            run.status, run.error = "failed", "服务重启导致运行中断 — 请用相同配置重建"
        return run


class RunManager:
    def __init__(self, runs_dir: Path, legacy_out: Path | None = None):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._index_cache: dict[str, dict] = {}
        for d in sorted(self.runs_dir.iterdir()):
            if d.is_dir() and (d / "run.json").exists():
                run = Run.load(d)
                if run:
                    self._runs[run.run_id] = run
        if legacy_out:
            self._import_legacy(Path(legacy_out))

    # ---- 查询 ----
    def list_runs(self) -> list[Run]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, run_id: str) -> Run | None:
        if run_id == "latest":  # 'latest' 别名 = 最近完成的运行 (前端适配层用)
            done = [r for r in self.list_runs() if r.status == "done"]
            return done[0] if done else None
        return self._runs.get(run_id)

    def load_index(self, run: Run) -> dict | None:
        """运行的 index.json (回合索引 + 预聚合); 完成运行不可变 → 进程内缓存。"""
        cached = self._index_cache.get(run.run_id)
        if cached is not None:
            return cached
        path = run.dir / "index.json"
        if not path.exists():
            return None
        index = json.loads(path.read_text())
        self._index_cache[run.run_id] = index
        return index

    def find_episode(self, episode_id: str) -> Path | None:
        """episode_id → 单回合 JSON 文件 (新运行优先, id 含 seed 全局唯一)。"""
        for run in self.list_runs():
            p = run.dir / "episodes" / f"{episode_id}.json"
            if p.exists():
                return p
        return None

    # ---- 发起 / 取消 ----
    def start_run(self, cfg: RunConfig) -> Run:
        with self._lock:
            run_id = (f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                      f"-{len(self._runs):03d}")
            run = Run(run_id, cfg, self.runs_dir / run_id)
            self._runs[run_id] = run
        run.save_meta()
        run.thread = threading.Thread(target=self._execute, args=(run,), daemon=True)
        run.thread.start()
        return run

    def cancel(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if not run or run.status in TERMINAL:
            return False
        run.cancel_event.set()
        return True

    def _execute(self, run: Run):
        try:
            result = run_benchmark(run.cfg, on_event=run.emit,
                                   cancel_event=run.cancel_event)
            index = export_run_data(run.dir, result["results"],
                                    result["meta"], result["episodes"])
            self._index_cache[run.run_id] = index
            run.set_status("done")
        except RunCancelled:
            run.emit({"type": "cancelled", "done": run.done, "total": run.total})
            run.set_status("cancelled")
        except Exception:
            err = traceback.format_exc()
            run.emit({"type": "error", "message": err.strip().splitlines()[-1]})
            run.set_status("failed", error=err)

    # ---- 历史产物导入 ----
    def _import_legacy(self, out_dir: Path):
        """把 CLI 产物 out/ 注册为一条已完成运行 (M-FE1 → M-FE2 平滑过渡)。"""
        marker = "run-000-legacy-out"
        if marker in self._runs or not (out_dir / "episodes.jsonl").exists():
            return
        from ..schema import Episode
        report = json.loads((out_dir / "report.json").read_text())
        episodes = [Episode.from_dict(json.loads(line))
                    for line in (out_dir / "episodes.jsonl").read_text().splitlines()]
        meta = report["meta"]
        cfg = RunConfig(seed=meta.get("seed", 7))
        run = Run(marker, cfg, self.runs_dir / marker, status="running",
                  created_at=datetime.fromtimestamp(
                      (out_dir / "episodes.jsonl").stat().st_mtime
                  ).astimezone().isoformat(timespec="seconds"))
        run.done = run.total = len(episodes)
        index = export_run_data(run.dir, report["results"], meta, episodes)
        self._index_cache[marker] = index
        run.set_status("done")
        self._runs[marker] = run
