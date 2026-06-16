"""Frontend data-adaptation layer (fe-rq.md §8/§13): turn evaluation artifacts into the index and pre-aggregations the frontend can consume directly.

M-FE1: build_web_data.py calls this module to pre-split out/ artifacts into web/data/ static files;
M-FE2: the platform API calls export_run_data() to persist an isomorphic data pack per run (NFR-FE N2:
the list carries no large arrays; O-F3: per-episode standalone JSON loaded on demand).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .schema import Episode
from .metrics import peak_uncertainty, plan_margin_ratio, tracking_error, wilson_ci

LUX_EDGES = [0.30, 0.44, 0.58, 0.72, 0.86, 1.0001]  # C2 robustness-curve buckets


def index_record(ep: Episode) -> dict:
    """Episode index row: list-column fields + precomputed metrics, no large arrays (NFR-FE N2)."""
    pu = peak_uncertainty(ep)
    return {
        "episode_id": ep.episode_id,
        "model_id": ep.model.model_id,
        "hw_config_id": ep.robot.hw_config_id,
        "task_type": ep.scene.task_type.value,
        "tolerance_class": ep.scene.tolerance_class.value,
        "generation_method": ep.scene.generation_method.value,
        "parent_episode_id": ep.scene.parent_episode_id,
        "mr_id": ep.scene.mr_id,
        "lux": float(ep.scene.perturbation.get("lux_factor", 1.0)),
        "success": ep.outcome.success,
        "failure_phase": ep.outcome.failure_phase.value if ep.outcome.failure_phase else None,
        "failure_label": ep.outcome.failure_label,
        "attribution": ep.outcome.attribution.value if ep.outcome.attribution else None,
        "attribution_reason": ep.outcome.attribution_reason,
        "duration_s": ep.outcome.duration_s,
        "plan_margin_ratio": round(plan_margin_ratio(ep), 4),
        "e_track_steady_rms_mm": round(tracking_error(ep)["steady_rms_m"] * 1e3, 4),
        "peak_uncertainty": None if pu is None else round(pu, 4),
    }


def combo_key(rec):
    return f"{rec['model_id']} @ {rec['hw_config_id']}"


def build_robustness(records):
    """C2: per combo × lux bucket → SR + Wilson CI."""
    out = {}
    for rec in records:
        out.setdefault(combo_key(rec), []).append(rec)
    labels = [f"{LUX_EDGES[i]:.2f}–{min(LUX_EDGES[i+1],1.0):.2f}" for i in range(len(LUX_EDGES) - 1)]
    series = {}
    for combo, recs in out.items():
        rows = []
        for i in range(len(LUX_EDGES) - 1):
            inb = [r for r in recs if LUX_EDGES[i] <= r["lux"] < LUX_EDGES[i + 1]]
            n, k = len(inb), sum(1 for r in inb if r["success"])
            lo, hi = wilson_ci(k, n) if n else (0.0, 1.0)
            rows.append({"n": n, "sr": (k / n) if n else None,
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "lux_min": LUX_EDGES[i], "lux_max": min(LUX_EDGES[i + 1], 1.0)})
        series[combo] = rows
    return {"bucket_labels": labels, "series": series}


def build_sankey(records):
    """C3: failure → phase → attribution → rule that fired."""
    links = Counter()
    for r in records:
        if r["success"]:
            continue
        phase = f"Phase:{r['failure_phase'] or '?'}"
        attr = f"Attribution:{r['attribution'] or 'unattributed'}"
        reason = r["attribution_reason"] or ""
        rule = "Rule:" + (reason.split("]", 1)[-1].strip().split(":")[0] if "]" in reason else "?")
        links[("Failure", phase)] += 1
        links[(phase, attr)] += 1
        links[(attr, rule)] += 1
    nodes = sorted({n for pair in links for n in pair})
    return {"nodes": [{"name": n} for n in nodes],
            "links": [{"source": s, "target": t, "value": v} for (s, t), v in sorted(links.items())]}


def build_failure_hist(records):
    """C5: first-failure phase × combo."""
    phases = ["grasp", "insert"]
    out = {}
    for r in records:
        if r["success"] or not r["failure_phase"]:
            continue
        out.setdefault(combo_key(r), Counter())[r["failure_phase"]] += 1
    return {"phases": phases,
            "series": {c: [cnt.get(p, 0) for p in phases] for c, cnt in sorted(out.items())}}


def build_radar(report):
    """C1: 6 axes, raw values + min-max normalization (baseline = all combos of this run, fe-rq.md C1)."""
    axes = ["Success rate", "Alignment-margin health", "Trajectory smoothness",
            "Tracking health", "Low jitter", "Uncertainty AUROC"]
    combos = []
    for r in report["results"]:
        raw = [
            r["sr"],
            1.0 / (1.0 + r["plan_margin_mean"]),
            1.0 / (1.0 + math.log10(1.0 + r["jerk_cmd_median"])),
            1.0 / (1.0 + r["e_track_rms_mean_mm"]),
            1.0 / (1.0 + math.log10(1.0 + r["jitter_band_mean"])),
            r["uncertainty_auroc"],  # None ⇒ frontend marks N/A (FR-FE-5.2)
        ]
        combos.append({"name": f"{r['model_id']} @ {r['hw_config_id']}", "raw": raw})
    norm_axes = []
    for i in range(len(axes)):
        vals = [c["raw"][i] for c in combos if c["raw"][i] is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        norm_axes.append((lo, hi))
    for c in combos:
        c["norm"] = [None if v is None else
                     (1.0 if hi == lo else round((v - lo) / (hi - lo), 4))
                     for v, (lo, hi) in zip(c["raw"], norm_axes)]
        c["raw"] = [None if v is None else round(v, 4) for v in c["raw"]]
    return {"axes": axes, "combos": combos}


def build_index(report: dict, records: list[dict]) -> dict:
    """index.json: meta + episode index + chart pre-aggregations (C1/C2/C3/C5, aggregated on the backend, §8)."""
    return {
        "meta": report["meta"],
        "episodes": records,
        "aggregates": {
            "radar": build_radar(report),
            "robustness": build_robustness(records),
            "sankey": build_sankey(records),
            "failure_hist": build_failure_hist(records),
        },
    }


def export_run_data(dest: Path, results: list[dict], meta: dict,
                    episodes: list[Episode]) -> dict:
    """Persist one run as a frontend data pack (M-FE2 one per run, same directory structure as web/data/):

      dest/report.json          run-level aggregation
      dest/index.json           episode index + pre-aggregations
      dest/episodes.jsonl       raw artifacts (for export / re-run)
      dest/episodes/<id>.json   full per-episode payload (loaded on demand by the debug page)

    Returns the index dict (the caller can put it straight into an in-memory cache).
    """
    dest = Path(dest)
    ep_dir = dest / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    report = {"meta": meta, "results": results}
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    records = []
    with open(dest / "episodes.jsonl", "w") as f:
        for ep in episodes:
            d = ep.to_dict()
            f.write(json.dumps(d, sort_keys=True) + "\n")
            records.append(index_record(ep))
            (ep_dir / f"{ep.episode_id}.json").write_text(json.dumps(d))

    index = build_index(report, records)
    (dest / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    return index
