#!/usr/bin/env python
"""Attribution threshold sensitivity: recompute attribution accuracy over an e_track-threshold × plan_margin-threshold grid.
Addresses the reviewer concern: thresholds are set a priori (attr-rules-0.1) — does the conclusion depend on fine-tuning?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pabench.scenegen import MutationGenerator, nominal_screw_cap
from pabench.models import PreciseVLA, SloppyVLA
from pabench.runners import CALIBRATED_ARM, WORN_ARM, FakeSimBackend
from pabench.attribution import AttributionThresholds, attribute_episode

backend = FakeSimBackend()
base = nominal_screw_cap()


def episodes(model, hw):
    eps = []
    for s in range(10):
        for i, sc in enumerate(MutationGenerator(seed=s).generate(base, "anchor", 100)):
            eps.append(backend.run_episode(sc, model, hw, seed=s * 100_000 + i))
    return eps


S_cal = episodes(SloppyVLA(), CALIBRATED_ARM)     # GT=model
P_worn = episodes(PreciseVLA(), WORN_ARM)         # GT=hardware

grid = {}
for tr_mm in [0.15, 0.20, 0.30, 0.45, 0.60]:
    for pm in [0.8, 1.0, 1.2]:
        th = AttributionThresholds(version=f"sens-{tr_mm}-{pm}",
                                   plan_margin_fail=pm, e_track_rms_max_m=tr_mm * 1e-3)
        ok = tot = 0
        for eps, gt, hw in [(S_cal, "model", CALIBRATED_ARM), (P_worn, "hardware", WORN_ARM)]:
            for ep in eps:
                if ep.outcome.success:
                    continue
                a = attribute_episode(
                    ep, th, oracle_fn=lambda e, _hw=hw: backend.run_oracle(
                        e.scene, _hw, int(e.episode_id.rsplit("__s", 1)[1])).outcome.success)
                tot += 1
                ok += (a.value == gt)
        grid[f"track={tr_mm}mm,plan={pm}"] = round(ok / tot, 4)

vals = list(grid.values())
res = {"grid": grid, "min": min(vals), "max": max(vals),
       "default": grid["track=0.3mm,plan=1.0"]}
Path(__file__).with_name("threshold_sensitivity.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
