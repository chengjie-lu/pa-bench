"""FR-6 可视化 (纵切版) —— 真实现: 静态 HTML 报告。
三层信息架构的第①层 (红绿灯摘要) + 第②层 (对比表/归因分布)。
第③层 (视频+3D 轨迹回放) 需要渲染管线, 列入已知限制; Web 交互平台为 M4。
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
    """results: 每个 (model × hw) 组合一条聚合记录 (见 demo.py summarize)。"""
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
            f"<td>{'❌ 违反' if r['mr1_violated'] else '✅ 通过'} "
            f"({r['mr1_median_dist_mm']:.1f}mm)</td>"
            f"<td>{ff}</td><td>{attr}</td></tr>")

    best = max(results, key=lambda x: x["sr"])
    summary = (f"<p class='big'>{_light(best['sr'])} 当前最优组合: "
               f"<b>{html.escape(best['model_id'])}</b> @ {html.escape(best['hw_config_id'])}, "
               f"成功率 {best['sr']:.0%}。绿=可用(≥80%), 黄=有风险(≥50%), 红=不可用。</p>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>PA-Bench 报告</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:2em;max-width:1200px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ccc;padding:6px 8px;text-align:left}}
th{{background:#f0f0f0}} .big{{font-size:18px}}
.meta{{color:#666;font-size:12px}}
</style></head><body>
<h1>PA-Bench 评测报告 — 瓶盖拧紧 (screw_cap, T1)</h1>
<p class="meta">benchmark_version: {html.escape(meta['benchmark_version'])} ·
seed: {meta['seed']} · 总回合数: {meta['total_episodes']} · 归因规则: {html.escape(meta['attr_rules_version'])}</p>
<h2>① 摘要 (面向非机器人背景读者)</h2>
{summary}
<h2>② 模型 × 硬件 对比 (工程层)</h2>
<table>
<tr><th></th><th>模型</th><th>硬件档案</th><th>成功率 [95% CI]</th>
<th>e_plan 裕度比均值<br>(>1=模型规划必败)</th><th>e_track RMS 均值 mm<br>(硬件跟踪)</th>
<th>抖动频段能量<br>(m/s²)²</th><th>指令 jerk 中位数</th><th>不确定性 AUROC</th>
<th>MR-1 等变性<br>(中位距离)</th><th>首败阶段分布</th><th>失败归因分布</th></tr>
{''.join(rows)}
</table>
<h2>③ 读法 (归因 → 改进动作)</h2>
<ul>
<li>归因 <b>model</b> 多 ⇒ 看 e_plan 裕度比与 MR 违反率: 偏置型 ⇒ 手眼标定/视觉微调; 随机型 ⇒ 补训练数据。</li>
<li>归因 <b>hardware</b> 多 ⇒ 看 e_track RMS 与抖动能量: 安排控制器整定或维保。</li>
<li>归因 <b>environment</b> ⇒ 扰动出训练分布且 oracle 也失败: 改工位条件或扩采集范围。</li>
</ul>
</body></html>"""


def write_report(path, results: list[dict], meta: dict):
    with open(path, "w") as f:
        f.write(build_html(results, meta))


def write_json(path, results: list[dict], meta: dict):
    with open(path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)