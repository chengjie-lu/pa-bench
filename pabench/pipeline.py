"""M-FE2 共享编排层 (fe-rq.md §13): demo CLI 与 platform API 复用同一条评测流水线。

职责: 场景生成 → (模型 × 硬件) 逐组合执行 → MR-1 协议判定 → 归因 → 组合摘要。
与 demo.py 原实现逐 seed 等价 (NFR-1: 同 RunConfig 产物哈希一致), 额外提供:
  - on_event 回调: 每回合/每组合进度事件 (SSE 推送的数据源, FR-FE-3.1)
  - cancel_event: 回合边界响应取消 (§4.3 取消运行)
  - pace_s: 演示节流 (FakeSim 全程 ~1s, 进度可视化时按需放慢)
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

# 可选对象注册表 (GET /api/models / /api/hardware 的数据源)。
# 接真实 VLA / 真机档案时在此登记即可被向导发现 (FR-2.4 / NFR-5)。
MODEL_REGISTRY = {m.model_id: m for m in (PreciseVLA, SloppyVLA)}
HW_REGISTRY = {hw.hw_config_id: hw for hw in (CALIBRATED_ARM, WORN_ARM)}


class RunCancelled(Exception):
    """cancel_event 置位后在回合边界抛出, 由调用方收尾。"""


@dataclass
class RunConfig:
    """一次评测运行的全部参数 (= /runs/new 向导三步的启动体, fe-rq.md §4.2)。"""
    model_ids: tuple = ("precise-vla-0.3", "sloppy-vla-0.1")
    hw_ids: tuple = ("arm-calibrated-2026Q2", "arm-worn-2023Q1")
    backend: str = "fake"  # 执行后端: fake | mujoco | agx (NFR-5 即插即用)
    seed: int = 7
    # 策略开关与预算 (步骤②): nominal 必跑 MR 才有源回合
    nominal: bool = True
    mutation_episodes: int = 24
    metamorphic: bool = True
    # MutationGenerator 形参一一对应 (§4.2 mutation 参数面板)
    pos_range_m: float = 0.015
    yaw_range_rad: float = 0.3
    lux_range: tuple = (0.3, 1.0)
    friction_range: tuple = (0.6, 1.2)
    pace_s: float = 0.0  # 每回合后 sleep, 仅演示用

    def validate(self):
        unknown = [m for m in self.model_ids if m not in MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"未注册的模型: {unknown}")
        unknown = [h for h in self.hw_ids if h not in HW_REGISTRY]
        if unknown:
            raise ValueError(f"未注册的硬件档案: {unknown}")
        if self.backend not in BACKEND_IDS:
            raise ValueError(f"未知后端: {self.backend!r} (可选: {BACKEND_IDS})")
        if not self.model_ids or not self.hw_ids:
            raise ValueError("至少各选 1 个模型与硬件档案")
        if self.mutation_episodes < 0:
            raise ValueError("mutation_episodes 不能为负")
        if self.metamorphic and not self.nominal:
            raise ValueError("MR-1 需要 nominal 源回合 (FR-1.3), 不能单独启用 metamorphic")
        if not self.nominal and self.mutation_episodes == 0:
            raise ValueError("至少启用一种场景策略")

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
    """场景生成 (在所有组合间共享 → 公平对比, NFR-2)。返回 (scenes, mrs)。"""
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
    """跑一个 (模型 × 硬件) 组合: 主回合 + MR-1 后继回合 + 归因。

    与 demo.py 原实现 seed 语义一致; on_episode(ep) 在每回合落库后调用。
    """
    def _tick(ep):
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled
        if on_episode:
            on_episode(ep)
        if pace_s:
            time.sleep(pace_s)

    store = EpisodeStore()
    # 主回合 (nominal + mutations)
    for i, scene in enumerate(scenes):
        ep = backend.run_episode(scene, model, hw, seed=seed_base + i)
        store.add(ep)
        _tick(ep)
    nominal_ep = store.episodes[0]

    # 变质测试 MR-1 (FR-1.3): 源回合 = nominal, 多旋转后继 → 协议级中位数判定
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

    # 归因 (FR-4): oracle 对照实验只在规则判不清时触发 (D1)
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
    """组合级摘要 (= report.json results[] 的一行)。"""
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
    """端到端执行一次评测运行。

    on_event(dict) 事件流 (与 SSE 事件一一对应, fe-rq.md §8):
      scenes_ready {mutations, mrs} → combo_start {combo, index, total_combos}
      → episode_done {combo, episode_id, success, failure_label, done, total}
      → combo_done {combo, summary} → run_done {total}
    取消时抛 RunCancelled (已完成回合不回滚, 由调用方决定保留与否)。
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
            # seed 语义与 demo.py 原实现一致 (NFR-1 跨入口可复现)
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
        merged.episodes.extend(s.episodes)  # 已校验过谱系, 直接合并
    meta = {"benchmark_version": BENCHMARK_VERSION, "seed": cfg.seed,
            "total_episodes": len(merged), "attr_rules_version": thresholds.version}
    emit({"type": "run_done", "total": len(merged)})
    return {"results": results, "meta": meta, "episodes": merged.episodes,
            "scenes": scenes, "models": models, "hws": hws}
