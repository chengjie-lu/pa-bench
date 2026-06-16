#!/usr/bin/env python
"""TSE paper experiment script: evaluate PA-Bench's discriminative power (RQ1) / attribution (RQ2) / actionability (RQ3) using fault-injection ground truth.

All statistics (two-proportion z-test, Cohen's h, Mann-Whitney U, Cliff's delta, bootstrap CI)
are implemented in numpy / pure python. Outputs JSON for citation in the paper.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pabench.schema import Attribution
from pabench.scenegen import MR1RotationZ, MutationGenerator, mr_violation_verdict, nominal_screw_cap
from pabench.models import PreciseVLA, SloppyVLA
from pabench.runners import CALIBRATED_ARM, WORN_ARM, FakeSimBackend
from pabench.runners.base import HardwareProfile
from pabench.metrics import jerk_cmd, plan_margin_ratio, tracking_error, wilson_ci
from pabench.metrics.l2_process import auroc, peak_uncertainty
from pabench.attribution import AttributionThresholds, attribute_episode

# ---------------- "improved" subjects matching a repair action (RQ3 fix-congruence) ----------------


class SloppyDebiasedVLA(SloppyVLA):
    """Model-side fix: hand-eye recalibration → removes the world-frame bias (noise/jitter kept)."""
    model_id = "sloppy-vla-0.1+debias"
    bias_world_m = np.zeros(2)


class PreciseSharpVLA(PreciseVLA):
    """Model-side fix (used as a wrong-fix control): perception noise halved."""
    model_id = "precise-vla-0.3+sharp"
    sigma_base_m = 0.175e-3


PRISTINE_ARM = HardwareProfile(  # hardware-side fix (used as a wrong-fix control): a brand-new calibrated arm
    hw_config_id="arm-pristine", tracking_alpha=0.5,
    droop_xy=(0.02e-3, -0.01e-3), track_noise_std=0.03e-3, jitter_amp=0.02e-3)

# ---------------- statistics utilities ----------------


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    pval = 2 * (1 - phi(abs(z)))
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))  # Cohen's h

    return {"p1": p1, "p2": p2, "z": z, "p_value": pval, "cohens_h": h}


def mann_whitney(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # tie-averaged ranks
    sv = allv[order]; i = 0
    rs = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        rs[i:j + 1] = rs[i:j + 1].mean(); i = j + 1
    ranks[order] = rs
    U = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mu) / sd if sd > 0 else 0.0
    pval = 2 * (1 - phi(abs(z)))
    delta = 2 * U / (n1 * n2) - 1  # Cliff's delta
    return {"U": U, "z": z, "p_value": pval, "cliffs_delta": delta}


def bootstrap_auroc(scores, labels, B=500, seed=0):
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores), np.asarray(labels)
    point = auroc(scores, labels)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(scores), len(scores))
        v = auroc(scores[idx], labels[idx])
        if v is not None:
            vals.append(v)
    return {"auroc": point, "ci95": [float(np.percentile(vals, 2.5)),
                                     float(np.percentile(vals, 97.5))]}


# ---------------- experiment execution ----------------

N_SEEDS, N_SCENES = 10, 100
backend = FakeSimBackend()
TH = AttributionThresholds()
base = nominal_screw_cap()


def run_condition(model, hw, attribute=False):
    """N_SEEDS × N_SCENES episodes; scenes are shared across conditions within the same seed (fair comparison)."""
    eps = []
    for s in range(N_SEEDS):
        scenes = MutationGenerator(seed=s).generate(base, "anchor", N_SCENES)
        for i, sc in enumerate(scenes):
            ep = backend.run_episode(sc, model, hw, seed=s * 100_000 + i)
            if attribute:
                attribute_episode(ep, TH, oracle_fn=lambda e, _hw=hw: backend.run_oracle(
                    e.scene, _hw, int(e.episode_id.rsplit("__s", 1)[1])).outcome.success)
            eps.append(ep)
    return eps


def sr_of(eps):
    k = sum(1 for e in eps if e.outcome.success)
    lo, hi = wilson_ci(k, len(eps))
    return {"k": k, "n": len(eps), "sr": k / len(eps), "ci95": [lo, hi]}


t0 = time.time()
res = {"config": {"n_seeds": N_SEEDS, "n_scenes": N_SCENES}}

print("== RQ1: discriminative power ==", flush=True)
P_cal = run_condition(PreciseVLA(), CALIBRATED_ARM, attribute=True)
S_cal = run_condition(SloppyVLA(), CALIBRATED_ARM, attribute=True)
P_worn = run_condition(PreciseVLA(), WORN_ARM, attribute=True)
S_worn = run_condition(SloppyVLA(), WORN_ARM)

sr_pc, sr_sc, sr_pw, sr_sw = sr_of(P_cal), sr_of(S_cal), sr_of(P_worn), sr_of(S_worn)
res["rq1"] = {
    "sr": {"precise_cal": sr_pc, "sloppy_cal": sr_sc,
           "precise_worn": sr_pw, "sloppy_worn": sr_sw},
    "model_contrast": two_prop_z(sr_pc["k"], sr_pc["n"], sr_sc["k"], sr_sc["n"]),
    "hw_contrast": two_prop_z(sr_pc["k"], sr_pc["n"], sr_pw["k"], sr_pw["n"]),
    "jerk_mw": mann_whitney([jerk_cmd(e) for e in S_cal], [jerk_cmd(e) for e in P_cal]),
    "plan_margin_mw": mann_whitney([plan_margin_ratio(e) for e in S_cal],
                                   [plan_margin_ratio(e) for e in P_cal]),
    "etrack_mw": mann_whitney([tracking_error(e)["steady_rms_m"] for e in P_worn],
                              [tracking_error(e)["steady_rms_m"] for e in P_cal]),
    # e_plan cross-hardware invariance (decomposition validity): precise cal vs worn
    "plan_margin_hw_invariance_mw": mann_whitney(
        [plan_margin_ratio(e) for e in P_worn], [plan_margin_ratio(e) for e in P_cal]),
}

print("== RQ2: attribution accuracy (fault-injection ground truth) ==", flush=True)


def attr_counts(eps):
    c = {}
    for e in eps:
        if not e.outcome.success and e.outcome.attribution:
            c[e.outcome.attribution.value] = c.get(e.outcome.attribution.value, 0) + 1
    return c


cm, ch = attr_counts(S_cal), attr_counts(P_worn)
n_m, n_h = sum(cm.values()), sum(ch.values())
acc_m = cm.get("model", 0) / n_m
acc_h = ch.get("hardware", 0) / n_h
res["rq2"] = {
    "model_injected": {"counts": cm, "n_failures": n_m, "accuracy": acc_m,
                       "ci95": list(wilson_ci(cm.get("model", 0), n_m))},
    "hw_injected": {"counts": ch, "n_failures": n_h, "accuracy": acc_h,
                    "ci95": list(wilson_ci(ch.get("hardware", 0), n_h))},
    "overall_accuracy": (cm.get("model", 0) + ch.get("hardware", 0)) / (n_m + n_h),
    "overall_ci95": list(wilson_ci(cm.get("model", 0) + ch.get("hardware", 0), n_m + n_h)),
    "ambiguous_rate": (cm.get("ambiguous", 0) + ch.get("ambiguous", 0)) / (n_m + n_h),
}

# MR-1 detection (20 seeds)
mr_det = {"precise_violations": 0, "sloppy_violations": 0, "n": 20}
for s in range(20):
    for model, key in [(PreciseVLA(), "precise_violations"), (SloppyVLA(), "sloppy_violations")]:
        src = backend.run_episode(base, model, CALIBRATED_ARM, seed=900_000 + s * 100)
        checks = []
        for j, th_ in enumerate([np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]):
            mr = MR1RotationZ(th_)
            fol = backend.run_episode(mr.apply(base, src.episode_id), model,
                                      CALIBRATED_ARM, seed=900_000 + s * 100 + j + 1)
            checks.append(mr.check(src, fol))
        if mr_violation_verdict(checks)["violated"]:
            mr_det[key] += 1
res["rq2"]["mr1"] = mr_det

print("== RQ3: actionability ==", flush=True)
# 3a uncertainty-warning AUROC (precise, pooled over all hardware to include both failure types)
pool = P_cal + P_worn
scores = [peak_uncertainty(e) for e in pool]
labels = [not e.outcome.success for e in pool]
res["rq3"] = {"uncertainty": {
    "precise_cal_only": bootstrap_auroc([peak_uncertainty(e) for e in P_cal],
                                        [not e.outcome.success for e in P_cal], seed=1),
    "precise_pooled_hw": bootstrap_auroc(scores, labels, seed=2),
    "sloppy_cal": bootstrap_auroc([peak_uncertainty(e) for e in S_cal],
                                  [not e.outcome.success for e in S_cal], seed=3)}}

# 3b fix-congruence: repair suggested by attribution vs a wrong repair
S_cal_debias = run_condition(SloppyDebiasedVLA(), CALIBRATED_ARM)
S_pristine = run_condition(SloppyVLA(), PRISTINE_ARM)
P_cal_after = run_condition(PreciseVLA(), CALIBRATED_ARM)  # = repairing the worn hardware
P_worn_sharp = run_condition(PreciseSharpVLA(), WORN_ARM)
sr_sd, sr_sp = sr_of(S_cal_debias), sr_of(S_pristine)
sr_pa, sr_ps = sr_of(P_cal_after), sr_of(P_worn_sharp)
res["rq3"]["fix_congruence"] = {
    "model_fault": {  # sloppy@cal, attribution=model
        "baseline": sr_sc,
        "right_fix_debias": {**sr_sd, **two_prop_z(sr_sd["k"], sr_sd["n"], sr_sc["k"], sr_sc["n"])},
        "wrong_fix_hw": {**sr_sp, **two_prop_z(sr_sp["k"], sr_sp["n"], sr_sc["k"], sr_sc["n"])}},
    "hw_fault": {     # precise@worn, attribution=hardware
        "baseline": sr_pw,
        "right_fix_hw": {**sr_pa, **two_prop_z(sr_pa["k"], sr_pa["n"], sr_pw["k"], sr_pw["n"])},
        "wrong_fix_model": {**sr_ps, **two_prop_z(sr_ps["k"], sr_ps["n"], sr_pw["k"], sr_pw["n"])}},
}

total_eps = 8 * N_SEEDS * N_SCENES + 20 * 2 * 4
res["throughput"] = {"total_episodes_approx": total_eps,
                     "wall_s": round(time.time() - t0, 1),
                     "eps_per_s": round(total_eps / (time.time() - t0), 1)}

out = Path(__file__).parent / "paper_results.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1)[:4000])
print("...\nsaved ->", out)
