#!/usr/bin/env python
"""跨后端排序一致性实验 (rq.md G4 思路的可执行替代):
8 个 (模型变体 × 硬件) 条件在 解析后端 vs MuJoCo 物理后端 上的 SR 排序 Spearman ρ。
需在 arm64 Python (mujoco 可用) 下运行。
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pabench.scenegen import MutationGenerator, nominal_screw_cap
from pabench.models import PreciseVLA, SloppyVLA
from pabench.runners import CALIBRATED_ARM, WORN_ARM, FakeSimBackend
from pabench.runners.mujoco_sim import MujocoBackend
from pabench.metrics import wilson_ci


class SloppyDebiasedVLA(SloppyVLA):
    model_id = "sloppy-vla-0.1+debias"
    bias_world_m = np.zeros(2)


class PreciseSharpVLA(PreciseVLA):
    model_id = "precise-vla-0.3+sharp"
    sigma_base_m = 0.175e-3


N_SEEDS, N_SCENES = 5, 60
MODELS = [PreciseVLA(), PreciseSharpVLA(), SloppyDebiasedVLA(), SloppyVLA()]
HWS = [CALIBRATED_ARM, WORN_ARM]
base = nominal_screw_cap()


def run_all(backend):
    out = {}
    for model in MODELS:
        for hw in HWS:
            k = n = 0
            for s in range(N_SEEDS):
                scenes = MutationGenerator(seed=s).generate(base, "anchor", N_SCENES)
                for i, sc in enumerate(scenes):
                    ep = backend.run_episode(sc, model, hw, seed=s * 100_000 + i)
                    k += ep.outcome.success
                    n += 1
            key = f"{model.model_id} @ {hw.hw_config_id}"
            lo, hi = wilson_ci(k, n)
            out[key] = {"sr": k / n, "n": n, "ci95": [lo, hi]}
    return out


def spearman_exact(a, b):
    """Spearman ρ + 精确置换 p 值 (n=8 → 40320 个置换)。"""
    def ranks(x):
        order = np.argsort(x)
        r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
        return r
    ra, rb = ranks(np.asarray(a)), ranks(np.asarray(b))
    def rho(x, y):
        x, y = x - x.mean(), y - y.mean()
        return float(np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y)))
    obs = rho(ra, rb)
    cnt = sum(1 for perm in itertools.permutations(rb)
              if rho(ra, np.array(perm)) >= obs)
    return obs, cnt / math.factorial(len(a))


t0 = time.time()
print("== 解析后端 ==", flush=True)
fake = run_all(FakeSimBackend())
print("== MuJoCo 后端 ==", flush=True)
mj = run_all(MujocoBackend())

keys = list(fake)
sr_f = [fake[k]["sr"] for k in keys]
sr_m = [mj[k]["sr"] for k in keys]
rho, pval = spearman_exact(sr_f, sr_m)
pearson = float(np.corrcoef(sr_f, sr_m)[0, 1])

res = {"config": {"n_seeds": N_SEEDS, "n_scenes": N_SCENES,
                  "episodes_per_cell": N_SEEDS * N_SCENES,
                  "n_conditions": len(keys)},
       "conditions": {k: {"fake_sr": fake[k]["sr"], "fake_ci": fake[k]["ci95"],
                          "mujoco_sr": mj[k]["sr"], "mujoco_ci": mj[k]["ci95"]}
                      for k in keys},
       "spearman_rho": rho, "perm_p_one_sided": pval, "pearson_r": pearson,
       "wall_s": round(time.time() - t0, 1)}

out = Path(__file__).parent / "crossbackend_results.json"
out.write_text(json.dumps(res, indent=1))
for k in sorted(keys, key=lambda x: -fake[x]["sr"]):
    print(f"{k:55s} fake={fake[k]['sr']:.3f}  mujoco={mj[k]['sr']:.3f}")
print(f"\nSpearman rho={rho:.3f} (exact perm p={pval:.5f})  Pearson r={pearson:.3f}")
print("saved ->", out)
