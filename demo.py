#!/usr/bin/env python
"""PA-Bench M1 vertical-slice end-to-end demo (rq.md §11 M1):

  scene generation (nominal + mutation + metamorphic MR-1)
  → FakeSimBackend executes 2 models × 2 hardware profiles
  → L0/L2/L3 metrics → e_plan/e_track attribution (incl. oracle-replay control)
  → console summary + out/report.html + out/report.json + out/episodes.jsonl
  → NFR-1 reproducibility self-check (re-run the first episode with the same seed, compare content hash)

The orchestration lives in pabench/pipeline.py (from M-FE2 on, shared with the platform API).

Usage: python demo.py [--episodes 24] [--seed 7] [--out out]
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
    ap.add_argument("--episodes", type=int, default=24, help="number of mutation episodes per combo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default="fake", choices=("fake", "mujoco", "agx"),
                    help="execution backend: fake (analytic stub) | mujoco | agx (AGX Dynamics)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    print(f"== PA-Bench M1 vertical-slice demo · {BENCHMARK_VERSION} · seed={args.seed} ==\n")

    # 0) metric registry machine-check (R-8)
    validate_registry()
    print("[0] metric registry machine-check passed: every metric binds an improvement action (FR-5.1)")

    # 1-4) scene generation + execution + metrics + attribution (orchestrated by pabench.pipeline, seed-equivalent to the original demo)
    cfg = RunConfig(seed=args.seed, mutation_episodes=args.episodes, backend=args.backend)
    print(f"[1] scene generation: 1 nominal + {args.episodes} mutation + {len(MR_THETAS)} MR-1 follow-ups / combo")
    run = run_benchmark(cfg)
    results, meta, episodes = run["results"], run["meta"], run["episodes"]
    print(f"[2] execution done: {len(episodes)} episodes (backend={args.backend}, 100 Hz telemetry)")

    # console summary
    hdr = (f"{'Model':<18} {'Hardware':<22} {'SR [95%CI]':<20} {'e_plan margin':<14} "
           f"{'e_track mm':<11} {'AUROC':<7} {'MR-1':<14} attribution mix")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        ci = f"{r['sr']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"
        au = "N/A" if r["uncertainty_auroc"] is None else f"{r['uncertainty_auroc']:.2f}"
        mr = ("violated" if r["mr1_violated"] else "passed") + f"({r['mr1_median_dist_mm']:.1f}mm)"
        attr = ",".join(f"{k}:{v}" for k, v in sorted(r["attribution_counts"].items())) or "—"
        print(f"{r['model_id']:<18} {r['hw_config_id']:<22} {ci:<20} "
              f"{r['plan_margin_mean']:<14.2f} {r['e_track_rms_mean_mm']:<11.2f} "
              f"{au:<7} {mr:<14} {attr}")

    # 5) write artifacts to disk
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from pabench.report.html_report import write_json, write_report
    write_report(out / "report.html", results, meta)
    write_json(out / "report.json", results, meta)
    merged = EpisodeStore()
    merged.episodes.extend(episodes)  # lineage already validated, merge directly
    merged.save_jsonl(out / "episodes.jsonl")
    print(f"\n[3] wrote: {out/'report.html'} · {out/'report.json'} · "
          f"{out/'episodes.jsonl'} ({len(merged)} episodes)")

    # 6) NFR-1 reproducibility self-check: re-run the first combo's first episode with the same seed, hashes must match
    ref = episodes[0]
    replay = FakeSimBackend().run_episode(run["scenes"][0], run["models"][0],
                                          run["hws"][0], seed=args.seed + 100_000)
    h1, h2 = ref.content_hash(), replay.content_hash()
    assert h1 == h2, f"NFR-1 reproducibility failed: {h1} != {h2}"
    print(f"[4] NFR-1 reproducibility self-check passed: same-seed replay hash matches ({h1[:16]}…)")
    print("\n== demo end-to-end complete ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())