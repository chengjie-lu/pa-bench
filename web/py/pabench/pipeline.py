"""M-FE2 shared orchestration layer (fe-rq.md §13): the demo CLI and the platform API reuse the same evaluation pipeline.

Responsibilities: scene generation → per-combo (model × hardware) execution → MR-1 protocol verdict → attribution → combo summary.
Per-seed equivalent to the original demo.py implementation (NFR-1: same RunConfig → identical artifact hash), and additionally provides:
  - on_event callback: per-episode/per-combo progress events (the data source pushed over SSE, FR-FE-3.1)
  - cancel_event: responds to cancellation at episode boundaries (§4.3 cancel a run)
  - pace_s: demo throttle (FakeSim runs ~1s total; slow it down when visualizing progress)
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from .schema import BENCHMARK_VERSION, EpisodeStore
from .scenegen import (EXPERT_DURATION_S, MR1RotationZ, MutationGenerator,
                       mr_violation_verdict, nominal_screw_cap)
from .models import PreciseVLA, SloppyVLA
from .runners import BACKEND_IDS, CALIBRATED_ARM, WORN_ARM, FakeSimBackend, make_backend
from .metrics import (efficiency_score, first_failure_histogram, jerk_cmd,
                      jitter_band_power, latency_percentiles, plan_margin_ratio,
                      success_rate, tracking_error, uncertainty_failure_auroc)
from .attribution import AttributionThresholds, attribute_episode

MR_THETAS = [np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]

# Registry of selectable objects (the data source for GET /api/models / /api/hardware).
# Registering a real VLA / real-robot profile here makes it discoverable by the wizard (FR-2.4 / NFR-5).
MODEL_REGISTRY = {m.model_id: m for m in (PreciseVLA, SloppyVLA)}
HW_REGISTRY = {hw.hw_config_id: hw for hw in (CALIBRATED_ARM, WORN_ARM)}


class RunCancelled(Exception):
    """Raised at an episode boundary once cancel_event is set; the caller does the cleanup."""


@dataclass
class RunConfig:
    """All parameters of one evaluation run (= the launch body of the 3-step /runs/new wizard, fe-rq.md §4.2)."""
    model_ids: tuple = ("precise-vla-0.3", "sloppy-vla-0.1")
    hw_ids: tuple = ("arm-calibrated-2026Q2", "arm-worn-2023Q1")
    backend: str = "fake"  # execution backend: fake | mujoco | agx (NFR-5 plug-and-play)
    seed: int = 7
    # strategy switches and budget (step ②): nominal must run for MR to have a source episode
    nominal: bool = True
    mutation_episodes: int = 24
    metamorphic: bool = True
    # one-to-one with MutationGenerator parameters (§4.2 mutation parameter panel)
    pos_range_m: float = 0.015
    yaw_range_rad: float = 0.3
    lux_range: tuple = (0.3, 1.0)
    friction_range: tuple = (0.6, 1.2)
    pace_s: float = 0.0  # sleep after each episode, demo only

    def validate(self):
        unknown = [m for m in self.model_ids if m not in MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"unregistered models: {unknown}")
        unknown = [h for h in self.hw_ids if h not in HW_REGISTRY]
        if unknown:
            raise ValueError(f"unregistered hardware profiles: {unknown}")
        if self.backend not in BACKEND_IDS:
            raise ValueError(f"unknown backend: {self.backend!r} (choices: {BACKEND_IDS})")
        if not self.model_ids or not self.hw_ids:
            raise ValueError("select at least 1 model and 1 hardware profile")
        if self.mutation_episodes < 0:
            raise ValueError("mutation_episodes cannot be negative")
        if self.metamorphic and not self.nominal:
            raise ValueError("MR-1 needs a nominal source episode (FR-1.3); metamorphic cannot be enabled on its own")
        if not self.nominal and self.mutation_episodes == 0:
            raise ValueError("enable at least one scene strategy")

    def episodes_per_combo(self) -> int:
        return ((1 if self.nominal else 0) + self.mutation_episodes
                + (len(MR_THETAS) if self.metamorphic else 0))

    def total_episodes(self) -> int:
        return self.episodes_per_combo() * len(self.model_ids) * len(self.hw_ids)

    def to_dict(self) -> dict:
        return {
            "model_ids": list(self.model_ids), "hw_ids": list(self.hw_ids),
            "backend": self.backend,
            "seed": self.seed, "nominal": self.nominal,
            "mutation_episodes": self.mutation_episodes, "metamorphic": self.metamorphic,
            "pos_range_m": self.pos_range_m, "yaw_range_rad": self.yaw_range_rad,
            "lux_range": list(self.lux_range), "friction_range": list(self.friction_range),
            "pace_s": self.pace_s,
        }

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        cfg = RunConfig(
            model_ids=tuple(d.get("model_ids") or RunConfig.model_ids),
            hw_ids=tuple(d.get("hw_ids") or RunConfig.hw_ids),
            backend=str(d.get("backend", RunConfig.backend)),
            seed=int(d.get("seed", 7)),
            nominal=bool(d.get("nominal", True)),
            mutation_episodes=int(d.get("mutation_episodes", 24)),
            metamorphic=bool(d.get("metamorphic", True)),
            pos_range_m=float(d.get("pos_range_m", 0.015)),
            yaw_range_rad=float(d.get("yaw_range_rad", 0.3)),
            lux_range=tuple(d.get("lux_range", (0.3, 1.0))),
            friction_range=tuple(d.get("friction_range", (0.6, 1.2))),
            pace_s=float(d.get("pace_s", 0.0)),
        )
        cfg.validate()
        return cfg


def build_scenes(cfg: RunConfig):
    """Scene generation (shared across all combos → fair comparison, NFR-2). Returns (scenes, mrs)."""
    base = nominal_screw_cap()
    scenes = [base] if cfg.nominal else []
    if cfg.mutation_episodes:
        mut = MutationGenerator(seed=cfg.seed, pos_range_m=cfg.pos_range_m,
                                yaw_range_rad=cfg.yaw_range_rad,
                                lux_range=cfg.lux_range,
                                friction_range=cfg.friction_range)
        scenes += mut.generate(base, f"{base.scene_id}__anchor", cfg.mutation_episodes)
    mrs = [MR1RotationZ(theta) for theta in MR_THETAS] if cfg.metamorphic else []
    return scenes, mrs


def run_combo(backend, model, hw, scenes, mrs, seed_base, thresholds,
              on_episode=None, cancel_event=None, pace_s=0.0):
    """Run one (model × hardware) combo: main episodes + MR-1 follow-up episodes + attribution.

    seed semantics match the original demo.py implementation; on_episode(ep) is called after each episode is stored.
    """
    def _tick(ep):
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled
        if on_episode:
            on_episode(ep)
        if pace_s:
            time.sleep(pace_s)

    store = EpisodeStore()
    # main episodes (nominal + mutations)
    for i, scene in enumerate(scenes):
        ep = backend.run_episode(scene, model, hw, seed=seed_base + i)
        store.add(ep)
        _tick(ep)
    nominal_ep = store.episodes[0]

    # metamorphic test MR-1 (FR-1.3): source episode = nominal, multiple rotated follow-ups → protocol-level median verdict
    checks, follow_ids = [], []
    for j, mr in enumerate(mrs):
        follow_scene = mr.apply(nominal_ep.scene, nominal_ep.episode_id)
        follow = backend.run_episode(follow_scene, model, hw, seed=seed_base + 9000 + j)
        store.add(follow)
        checks.append(mr.check(nominal_ep, follow))
        follow_ids.append(follow.episode_id)
        _tick(follow)
    verdict = mr_violation_verdict(checks) if mrs else {"violated": False, "median_dist_m": 0.0}
    violated_ids = set(follow_ids) if verdict["violated"] else set()
    if verdict["violated"]:
        for ep in store.episodes:
            if ep.episode_id in violated_ids:
                ep.outcome.failure_label = (ep.outcome.failure_label or "") + "|mr1_violation"

    # attribution (FR-4): the oracle control experiment is triggered only when the rules are inconclusive (D1)
    oracle_calls = 0

    def oracle_fn(ep):
        nonlocal oracle_calls
        oracle_calls += 1
        seed = int(ep.episode_id.rsplit("__s", 1)[1])
        return backend.run_oracle(ep.scene, hw, seed).outcome.success

    for ep in store.episodes:
        attribute_episode(ep, thresholds,
                          mr_violated=ep.episode_id in violated_ids,
                          oracle_fn=oracle_fn)
    return store, verdict, oracle_calls


def summarize(model_id, hw_id, store, mr_verdict, oracle_calls):
    """Combo-level summary (= one row of report.json results[])."""
    eps = store.episodes
    sr = success_rate(eps)
    failed = [e for e in eps if not e.outcome.success]
    attr_counts = {}
    for e in failed:
        if e.outcome.attribution:
            k = e.outcome.attribution.value
            attr_counts[k] = attr_counts.get(k, 0) + 1
    auroc_v = uncertainty_failure_auroc(eps)
    return {
        "model_id": model_id, "hw_config_id": hw_id,
        "sr": sr["sr"], "n": sr["n"], "ci95": list(sr["ci95"]),
        "efficiency": efficiency_score(eps, EXPERT_DURATION_S),
        "plan_margin_mean": statistics.mean(plan_margin_ratio(e) for e in eps),
        "e_track_rms_mean_mm": statistics.mean(
            tracking_error(e)["steady_rms_m"] for e in eps) * 1e3,
        "jitter_band_mean": statistics.mean(jitter_band_power(e) for e in eps),
        "jerk_cmd_median": statistics.median(jerk_cmd(e) for e in eps),
        "uncertainty_auroc": auroc_v,
        "latency_p99_ms": statistics.mean(latency_percentiles(e)["p99_ms"] for e in eps),
        "mr1_violated": mr_verdict["violated"],
        "mr1_median_dist_mm": mr_verdict["median_dist_m"] * 1e3,
        "first_failure": first_failure_histogram(eps),
        "attribution_counts": attr_counts,
        "oracle_replays_used": oracle_calls,
    }


def run_benchmark(cfg: RunConfig, on_event=None, cancel_event=None) -> dict:
    """Run one evaluation run end to end.

    on_event(dict) event stream (one-to-one with the SSE events, fe-rq.md §8):
      scenes_ready {mutations, mrs} → combo_start {combo, index, total_combos}
      → episode_done {combo, episode_id, success, failure_label, done, total}
      → combo_done {combo, summary} → run_done {total}
    On cancellation it raises RunCancelled (completed episodes are not rolled back; the caller decides whether to keep them).
    """
    cfg.validate()
    emit = on_event or (lambda e: None)
    scenes, mrs = build_scenes(cfg)
    emit({"type": "scenes_ready", "mutations": cfg.mutation_episodes, "mrs": len(mrs)})

    backend = make_backend(cfg.backend)
    thresholds = AttributionThresholds()
    models = [MODEL_REGISTRY[m]() for m in cfg.model_ids]
    hws = [HW_REGISTRY[h] for h in cfg.hw_ids]
    total = cfg.total_episodes()
    total_combos = len(models) * len(hws)

    results, all_stores, done = [], [], 0
    for mi, model in enumerate(models):
        for hi, hw in enumerate(hws):
            combo_idx = mi * len(hws) + hi
            combo = f"{model.model_id} @ {hw.hw_config_id}"
            # seed semantics match the original demo.py implementation (NFR-1 reproducible across entry points)
            seed_base = cfg.seed + 100_000 * (combo_idx + 1)
            emit({"type": "combo_start", "combo": combo,
                  "index": combo_idx, "total_combos": total_combos})

            def on_episode(ep, combo=combo):
                nonlocal done
                done += 1
                emit({"type": "episode_done", "combo": combo,
                      "episode_id": ep.episode_id, "success": ep.outcome.success,
                      "failure_label": ep.outcome.failure_label,
                      "done": done, "total": total})

            store, mr_verdict, oracle_calls = run_combo(
                backend, model, hw, scenes, mrs, seed_base, thresholds,
                on_episode=on_episode, cancel_event=cancel_event, pace_s=cfg.pace_s)
            summary = summarize(model.model_id, hw.hw_config_id,
                                store, mr_verdict, oracle_calls)
            results.append(summary)
            all_stores.append(store)
            emit({"type": "combo_done", "combo": combo, "summary": summary})

    merged = EpisodeStore()
    for s in all_stores:
        merged.episodes.extend(s.episodes)  # lineage already validated, merge directly
    meta = {"benchmark_version": BENCHMARK_VERSION, "seed": cfg.seed,
            "total_episodes": len(merged), "attr_rules_version": thresholds.version}
    emit({"type": "run_done", "total": len(merged)})
    return {"results": results, "meta": meta, "episodes": merged.episodes,
            "scenes": scenes, "models": models, "hws": hws}
