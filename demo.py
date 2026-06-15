#!/usr/bin/env python
"""PA-Bench M1 纵切端到端 demo (rq.md §11 M1):

  场景生成 (nominal + 变异 + 变质 MR-1)
  → FakeSimBackend 执行 2 模型 × 2 硬件档位
  → L0/L2/L3 指标 → e_plan/e_track 归因 (含 oracle 回放对照)
  → 控制台摘要 + out/report.html + out/report.json + out/episodes.jsonl
  → NFR-1 复现自检 (同 seed 重跑首回合, 比对内容哈希)

编排逻辑在 pabench/pipeline.py (M-FE2 起与 platform API 共用同一条流水线)。

用法: python demo.py [--episodes 24] [--seed 7] [--out out]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pabench.schema import BENCHMARK_VERSION, EpisodeStore
from pabench.pipeline import MR_THETAS, RunConfig, run_benchmark
from pabench.runners import FakeSimBackend
from pabench.metrics import validate_registry


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24, help="每组合的变异回合数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default="fake", choices=("fake", "mujoco", "agx"),
                    help="执行后端: fake(解析桩) | mujoco | agx(AGX Dynamics)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    print(f"== PA-Bench M1 纵切 demo · {BENCHMARK_VERSION} · seed={args.seed} ==\n")

    # 0) 指标注册表机检 (R-8)
    validate_registry()
    print("[0] 指标注册表机检通过: 全部指标均绑定改进动作 (FR-5.1)")

    # 1-4) 场景生成 + 执行 + 指标 + 归因 (pabench.pipeline 编排, 与 demo 原实现 seed 等价)
    cfg = RunConfig(seed=args.seed, mutation_episodes=args.episodes, backend=args.backend)
    print(f"[1] 场景生成: 1 nominal + {args.episodes} mutation + {len(MR_THETAS)} MR-1 后继 / 组合")
    run = run_benchmark(cfg)
    results, meta, episodes = run["results"], run["meta"], run["episodes"]
    print(f"[2] 执行完成: {len(episodes)} 回合 (backend={args.backend}, 100 Hz 遥测)")

    # 控制台摘要
    hdr = (f"{'模型':<18} {'硬件':<22} {'SR [95%CI]':<20} {'e_plan裕度':<10} "
           f"{'e_track mm':<11} {'AUROC':<7} {'MR-1违反':<9} 归因分布")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        ci = f"{r['sr']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"
        au = "N/A" if r["uncertainty_auroc"] is None else f"{r['uncertainty_auroc']:.2f}"
        mr = ("违反" if r["mr1_violated"] else "通过") + f"({r['mr1_median_dist_mm']:.1f}mm)"
        attr = ",".join(f"{k}:{v}" for k, v in sorted(r["attribution_counts"].items())) or "—"
        print(f"{r['model_id']:<18} {r['hw_config_id']:<22} {ci:<20} "
              f"{r['plan_margin_mean']:<10.2f} {r['e_track_rms_mean_mm']:<11.2f} "
              f"{au:<7} {mr:<9} {attr}")

    # 5) 落盘
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from pabench.report.html_report import write_json, write_report
    write_report(out / "report.html", results, meta)
    write_json(out / "report.json", results, meta)
    merged = EpisodeStore()
    merged.episodes.extend(episodes)  # 已校验过谱系, 直接合并
    merged.save_jsonl(out / "episodes.jsonl")
    print(f"\n[3] 已写出: {out/'report.html'} · {out/'report.json'} · "
          f"{out/'episodes.jsonl'} ({len(merged)} 回合)")

    # 6) NFR-1 复现自检: 同 seed 重跑首组合首回合, 哈希必须一致
    ref = episodes[0]
    replay = FakeSimBackend().run_episode(run["scenes"][0], run["models"][0],
                                          run["hws"][0], seed=args.seed + 100_000)
    h1, h2 = ref.content_hash(), replay.content_hash()
    assert h1 == h2, f"NFR-1 复现失败: {h1} != {h2}"
    print(f"[4] NFR-1 复现自检通过: 同 seed 重放哈希一致 ({h1[:16]}…)")
    print("\n== demo 端到端完成 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
