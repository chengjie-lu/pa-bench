"""Run manager: launch / progress events / cancel / persist / load history for evaluation runs (fe-rq.md §4.3, §11).

One directory per run, runs/<run_id>/:
  run.json          run metadata (config + status + progress), written on every status change
  report.json       run-level aggregation (after completion)
  index.json        episode index + chart pre-aggregations (after completion, same structure as web/data/index.json)
  episodes.jsonl    raw artifacts
  episodes/<id>.json full single episode (loaded on demand by the debug page, NFR-FE N2)

Execution runs on a background thread (FakeSim is a CPU sequential simulation; a single worker is enough for the M1 slice);
the event history lives in memory + SSE waiters are woken via a Condition (FR-FE-3.1 reflects progress within ≤2s).
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
    """In-memory state of a single evaluation run (event history + progress counters + cancel handle)."""

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

    # ---- event stream (SSE data source) ----
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
        """Block until there is a new event or a terminal state is reached; returns whether there is a new event (used for SSE heartbeat decisions)."""
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

    # ---- persistence ----
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
        if run.status == "running":  # an orphan run after a service restart → failed (the thread no longer exists)
            run.status, run.error = "failed", "run interrupted by a service restart — please rebuild with the same config"
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

    # ---- queries ----
    def list_runs(self) -> list[Run]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, run_id: str) -> Run | None:
        if run_id == "latest":  # the 'latest' alias = the most recently completed run (used by the frontend adapter)
            done = [r for r in self.list_runs() if r.status == "done"]
            return done[0] if done else None
        return self._runs.get(run_id)

    def load_index(self, run: Run) -> dict | None:
        """The run's index.json (episode index + pre-aggregations); a completed run is immutable → in-process cache."""
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
        """episode_id → single-episode JSON file (newer runs first; the id contains the seed and is globally unique)."""
        for run in self.list_runs():
            p = run.dir / "episodes" / f"{episode_id}.json"
            if p.exists():
                return p
        return None

    # ---- launch / cancel ----
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

    # ---- import historical artifacts ----
    def _import_legacy(self, out_dir: Path):
        """Register the CLI artifacts in out/ as one completed run (smooth M-FE1 → M-FE2 transition)."""
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
