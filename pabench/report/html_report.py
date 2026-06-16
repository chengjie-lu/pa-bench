"""FR-6 visualization (vertical-slice version) — real implementation: a static HTML report.
Layer ① (traffic-light summary) + layer ② (comparison table / attribution mix) of the three-layer information architecture.
Layer ③ (video + 3D trajectory replay) needs a render pipeline, listed as a known limitation; the interactive web platform is M4.
"""
from __future__ import annotations

import html
import json


def _light(sr: float) -> str:
    if sr >= 0.8:
        return "🟢"
    if sr >= 0.5:
        return "🟡"
    return "🔴"


def _fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def build_html(results: list[dict], meta: dict) -> str:
    """results: one aggregate record per (model × hw) combo (see demo.py summarize)."""
    rows = []
    for r in sorted(results, key=lambda x: -x["sr"]):
        attr = ", ".join(f"{k}:{v}" for k, v in sorted(r["attribution_counts"].items())) or "—"
        ff = ", ".join(f"{k}:{v}" for k, v in sorted(r["first_failure"].items())) or "—"
        rows.append(
            "<tr>"
            f"<td>{_light(r['sr'])}</td><td>{html.escape(r['model_id'])}</td>"
            f"<td>{html.escape(r['hw_config_id'])}</td>"
            f"<td>{r['sr']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] (n={r['n']})</td>"
            f"<td>{_fmt(r['plan_margin_mean'])}</td>"
            f"<td>{_fmt(r['e_track_rms_mean_mm'])}</td>"
            f"<td>{_fmt(r['jitter_band_mean'])}</td>"
            f"<td>{_fmt(r['jerk_cmd_median'], 3)}</td>"
            f"<td>{_fmt(r['uncertainty_auroc'])}</td>"
            f"<td>{'❌ violated' if r['mr1_violated'] else '✅ passed'} "
            f"({r['mr1_median_dist_mm']:.1f}mm)</td>"
            f"<td>{ff}</td><td>{attr}</td></tr>")

    best = max(results, key=lambda x: x["sr"])
    summary = (f"<p class='big'>{_light(best['sr'])} Current best combo: "
               f"<b>{html.escape(best['model_id'])}</b> @ {html.escape(best['hw_config_id'])}, "
               f"success rate {best['sr']:.0%}. Green=usable(≥80%), yellow=at risk(≥50%), red=not usable.</p>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>PA-Bench report</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:2em;max-width:1200px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ccc;padding:6px 8px;text-align:left}}
th{{background:#f0f0f0}} .big{{font-size:18px}}
.meta{{color:#666;font-size:12px}}
</style></head><body>
<h1>PA-Bench evaluation report — screw cap (screw_cap, T1)</h1>
<p class="meta">benchmark_version: {html.escape(meta['benchmark_version'])} ·
seed: {meta['seed']} · total episodes: {meta['total_episodes']} · attribution rules: {html.escape(meta['attr_rules_version'])}</p>
<h2>① Summary (for readers without a robotics background)</h2>
{summary}
<h2>② Model × hardware comparison (engineering layer)</h2>
<table>
<tr><th></th><th>Model</th><th>Hardware profile</th><th>Success rate [95% CI]</th>
<th>Mean e_plan margin ratio<br>(>1 = model plan must fail)</th><th>Mean e_track RMS mm<br>(hardware tracking)</th>
<th>Jitter band energy<br>(m/s²)²</th><th>Median command jerk</th><th>Uncertainty AUROC</th>
<th>MR-1 equivariance<br>(median distance)</th><th>First-failure phase mix</th><th>Failure attribution mix</th></tr>
{''.join(rows)}
</table>
<h2>③ How to read it (attribution → improvement action)</h2>
<ul>
<li>Mostly <b>model</b> attribution ⇒ look at e_plan margin ratio and MR violation rate: bias-type ⇒ hand-eye calibration / vision fine-tune; random-type ⇒ add training data.</li>
<li>Mostly <b>hardware</b> attribution ⇒ look at e_track RMS and jitter energy: schedule controller tuning or maintenance.</li>
<li><b>environment</b> attribution ⇒ perturbation out of the training distribution and the oracle also fails: change the workstation conditions or widen the collection range.</li>
</ul>
</body></html>"""


def write_report(path, results: list[dict], meta: dict):
    with open(path, "w") as f:
        f.write(build_html(results, meta))


def write_json(path, results: list[dict], meta: dict):
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)