# PA-Bench — VLA Precision-Assembly Evaluation Benchmark (M1 vertical slice)

[![Live demo](https://img.shields.io/badge/live%20demo-pa--bench-2f63d8)](https://chengjie-lu.github.io/pa-bench/)
[![Deploy](https://github.com/chengjie-lu/pa-bench/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/chengjie-lu/pa-bench/actions/workflows/deploy-pages.yml)

**Live console: https://chengjie-lu.github.io/pa-bench/** — overview, run results, charts, episode browser, and the per-episode debug player work immediately from precomputed data. A few seconds after load it also boots an **in-browser Pyodide runtime** (CPython + numpy in WebAssembly) so you can **launch new evaluations and register custom metrics right on the live link, with no backend** — the brand chip flips to "in-browser (Pyodide)" once ready.

The M1 minimal closed loop (§11 of the requirements doc): an end-to-end evaluation chain for the single task
**cap fastening (screw_cap, T1)** — scene generation (nominal + mutation + metamorphic MR-1) → simulation
(2 models × 2 hardware profiles) → L0/L2/L3 three-layer metrics → e_plan/e_track failure attribution
(with an oracle-replay control experiment) → HTML/JSON report. The whole chain is reproducible under a fixed
seed (NFR-1, with a built-in hash self-check in the demo).

## Run it from scratch

```bash
# Environment: Python 3.11
pip install -r requirements.txt

# End-to-end demo (≈1s): console summary + out/report.html + out/report.json + out/episodes.jsonl
python demo.py --episodes 24 --seed 7 --out out
# pick a physics backend: --backend fake (default) | mujoco | agx

# Test suite
python -m pytest tests/ -q

# Web console A — M-FE1 static mode (no backend):
python build_web_data.py                    # out/ artifacts → web/data/ (index + pre-aggregation + per-episode split)
python -m http.server 8765 --directory web  # open http://localhost:8765/

# Web console B — M-FE2 server mode (FastAPI + SSE, launch runs from the browser):
python serve.py                             # open http://127.0.0.1:8000/
#   ① Runs → New run: 3-step wizard (models × hardware / strategy + budget / seed + launch)
#   ② after launch, watch live SSE progress → auto-redirect to results → drill into per-episode debug
#   existing out/ artifacts are auto-imported as the historical run run-000-legacy-out
```

> A single `web/app.js` runs in both modes: it probes `/api/ping` on startup — a hit means server mode
> (platform API + launch runs + SSE progress), otherwise it falls back to M-FE1 static mode (reads `web/data/`).

Web console views: ① Overview (traffic-light cards + Δ arrows + Top-3 weakness drill-down); ② Run results
(combo comparison table + radar / robustness curve / attribution Sankey / first-failure histogram, charts drill
down on click); ③ Episode browser (multi-dimensional filters, state written to the URL); ④ per-episode debug page
(command vs measured trajectory replay + tolerance magnifier + four synchronized timeline panels + linked playhead,
space to play/pause, ←/→ to switch episodes). Server mode adds: ② Runs (list + 3-step new-run wizard + SSE progress
page). ECharts is vendored (web/vendor/) and works offline.

## File tree

```
pa-bench/
├── demo.py                      # end-to-end CLI (slice entry point, thin wrapper → pabench.pipeline)
├── serve.py                     # M-FE2 server-mode entry (FastAPI + static frontend, one process)
├── build_web_data.py            # M-FE1 static adapter (thin wrapper → pabench.webdata)
├── requirements.txt
├── .github/workflows/           # GitHub Actions: build web data + deploy to GitHub Pages
├── pabench/
│   ├── schema.py                # ★ Episode core contract + lineage-checking EpisodeStore (rq.md §5, FR-1.6)
│   ├── pipeline.py              # ★ shared evaluation orchestration RunConfig/run_benchmark (shared by demo & API, progress callback + cancel)
│   ├── webdata.py               # frontend data adapter: episode index + C1/C2/C3/C5 pre-aggregation (NFR-FE N2)
│   ├── platform/                # M-FE2 service layer (note: subpackage avoids shadowing the stdlib platform module)
│   │   ├── run_manager.py       # run launch / progress events / cancel / persistence / history import (background thread + SSE source)
│   │   └── api.py               # FastAPI: all §8 endpoints + SSE + stride downsampling + static mount
│   ├── scenegen/
│   │   ├── nominal.py           # FR-1.1 nominal task (screw_cap T1) + phase plan
│   │   ├── mutation.py          # FR-1.2 mutation generation (pose/lighting/friction, all params recorded)
│   │   └── metamorphic.py       # FR-1.3 MR-1 rotational equivariance + protocol-level median verdict
│   ├── models/
│   │   ├── base.py              # FR-2.4 VLAModel standard interface (uncertainty optional)
│   │   └── fake.py              # [FAKE] 2 scripted fake models (precise / sloppy)
│   ├── runners/
│   │   ├── base.py              # Backend abstraction + HardwareProfile (hardware calibration profile)
│   │   ├── fake_sim.py          # [FAKE backend] analytic-kinematics fake sim + oracle replay (FR-2.5)
│   │   ├── mujoco_sim.py        # MuJoCo physics backend, graceful degradation when import fails
│   │   └── agx_sim.py           # AGX Dynamics physics backend (industrial, licensed), graceful degradation
│   ├── metrics/
│   │   ├── l0_outcome.py        # SR + Wilson CI, efficiency score, first-failure distribution (FR-3.1/3.2/3.4)
│   │   ├── l2_process.py        # alignment residual / jerk / force-exceed / uncertainty AUROC / latency (FR-3.5–3.10)
│   │   ├── l3_hardware.py       # e_track RMS (steady-state), 5–50Hz jitter PSD (FR-3.11/3.12)
│   │   └── registry.py          # FR-5.1 metric registry + R-8 machine-check (no improvement action → cannot ship)
│   ├── attribution/engine.py    # FR-4 attribution decision tree + oracle control-experiment orchestration
│   └── report/html_report.py    # FR-6 slice version: traffic-light summary + engineering comparison table (static HTML)
├── web/                         # frontend console (native ES modules + ECharts, M-FE1 static / M-FE2 server dual mode)
└── tests/                       # behavioral-assertion tests (incl. test_api.py platform API)
```

## Real implementation vs fake (stub)

| Part | Status | Notes |
|---|---|---|
| Episode schema / lineage check / hash reproducibility | ✅ real | system-wide contract |
| Scene generation (mutation / MR-1 / protocol verdict) | ✅ real | MR-2/3/4 not done (see below) |
| Metric computation (10 across L0/L2/L3) + registry machine-check | ✅ real | Wilson/AUROC/PSD/jerk all hand-written in numpy with formula-level unit tests |
| Attribution decision tree + oracle control orchestration | ✅ real | versioned thresholds `attr-rules-0.1` |
| HTML/JSON report | ✅ real | the first two layers of the three-layer information architecture |
| Web console (M-FE1 static + M-FE2 server) | ✅ real | FastAPI + SSE progress + browser-launched runs; one frontend, dual mode |
| VLA models | 🔶 **FAKE** `pabench/models/fake.py` | scripted, behavior-controllable fake models |
| Simulation backend | 🔶 **FAKE** `pabench/runners/fake_sim.py` | analytic-kinematics approximate physics |
| MuJoCo backend | ✅ real (optional dep) `pabench/runners/mujoco_sim.py` | graceful degradation + clear guidance when mujoco is absent |
| AGX Dynamics backend | ✅ real (licensed dep) `pabench/runners/agx_sim.py` | graceful degradation when agx is absent/unlicensed |

**Files to change to wire up a real backend:**
1. Real VLA → add `pabench/models/<your_model>.py` implementing `VLAModel.infer()` (calling gRPC/HTTP internally); nothing else changes.
2. Real simulation → MuJoCo and AGX backends already implement the same `Backend` interface; select with `RunConfig.backend` / `--backend`. Add another by subclassing `Backend`.
3. Real robot → add `pabench/runners/real_robot.py` implementing the same `Backend` interface (plus FR-2.2 init checks and the NFR-4 safety loop, not covered by this slice).

## Assumptions & to-confirm

- Mutation scenes are shared across all (model × hardware) combos for fair comparison (NFR-2); their `parent_episode_id`
  points to the nominal scene's anchor record rather than a specific episode (the nominal episode id varies per combo).
- The fake physics error chain is deliberately constructed to be decomposable into "perception error (→e_plan) + tracking
  error (→e_track)", to validate the attribution mechanism itself; under real simulation/robot the thresholds need
  recalibration from the first batch of data per rq.md O-6.
- The e_track used for attribution is the RMS over the fasten steady-state window (so motion-segment tracking lag does not contaminate the hardware discriminator).
- The MR verdict uses time-synchronized point-wise distance rather than DTW (DTW's time warping would absorb a bias along the path direction and weaken non-equivariance detection; DTW is reserved for future unequal-length trajectories).
- The M-FE2 service layer lives in `pabench/platform/` rather than the repo-top-level `platform/` suggested by fe-rq.md O-F4:
  a top-level directory of that name would shadow Python's stdlib `platform` module (numpy and others import it at startup).
- Run execution uses a background thread rather than a process/task queue (FakeSim is a CPU sequential simulation; a single
  worker is enough for the M1 slice); swap in the §8 task queue when wiring up a real/heavy backend. On a service restart,
  orphaned running runs are marked failed.

## Known limitations & next steps (per the rq.md milestones)

1. **M1 remainder**: MR-2/3/4 metamorphic relations; closed-loop chunked inference (the model currently emits the whole trajectory at once).
2. **M2**: real-robot Runner + safety loop; FR-3.13 repeatability / FR-3.14 friction identification (need a real-robot calibration procedure); expert blind-label calibration of attribution (FR-4.4).
3. **M3**: FR-1.4 adversarial/optimization search (CMA-ES failure-boundary exploration); FR-1.5 real-robot stratified sampling; sim-real ranking-consistency measurement (G4).
4. **M4**: task-matrix expansion (5 task types × 3 tolerance classes); FR-5.2 targeted-collection checklist.

### Frontend milestones (fe-rq.md §13)

- **M-FE1 static mode** ✅: overview / run results / episode browser / per-episode debug + C1/C2/C3/C5 charts, reading `web/data/`.
- **M-FE2 service mode** ✅: FastAPI wrapper + `/runs/new` 3-step wizard (no scene editor) + live SSE progress +
  run list/detail/cancel/rebuild + legacy `out/` import. Two API-launched runs with the same seed produce identical artifacts (FR-FE-2.1).
  The wizard also picks the **execution backend** (fake / mujoco / agx; unavailable ones are shown disabled).
  **User-registered metrics** (FR-5.1): the Metric registry page has a registration form where users add a metric as a
  safe per-episode formula over whitelisted fields (`success`, `plan_margin_ratio`, `e_track_steady_rms_mm`,
  `peak_uncertainty`, `lux`, `duration_s`) + an aggregation (mean/median/max/min/sum/rate). Formulas are evaluated by an
  AST-hardened evaluator (no `eval`); R-8 still requires each to bind ≥1 improvement action. Registered metrics persist
  (`custom_metrics.json`), show in the registry, and appear as extra columns in the run-results combo table — computed
  on demand so they apply to existing runs too. API: `GET /api/backends`, `GET /api/metric-fields`, `GET/POST/DELETE /api/custom-metrics`.
- **In-browser runtime (Pyodide)** ✅: on the static deploy (no backend), a Web Worker loads Pyodide + numpy,
  imports the pabench sources emitted to `web/py/` by `build_web_data.py`, and serves an in-memory backend
  (`pabench/browser_api.py`) through a `/api/*` `fetch` interceptor. So the run wizard and custom-metric
  registration work on the public link exactly as in server mode; runs execute client-side (synchronously, ~1s).
  Falls back silently to view-only static mode if WebAssembly/CDN is unavailable.
- **M-FE3 interactive scenes** ⬜: scene-editor mouse/keyboard interaction (drag pose / adjust lighting), oracle-control button, side-by-side comparison of two runs.
- **M-FE4 customer-facing** ⬜: PDF report export, `/hardware` hardware-trend page, R-9 non-expert user testing.
- Degradation clauses still apply: the 3D viewport is a 2D top-down canvas (N5); FakeSim does not render, so the debug-page video area is always a placeholder (O-F2).