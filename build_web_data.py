#!/usr/bin/env python
"""M-FE1 静态适配层 (fe-rq.md §13): 把 out/ 产物预拆分/预聚合成前端可直接 fetch 的静态数据。

构建逻辑在 pabench/webdata.py (M-FE2 起与 platform API 共用)。

输入: out/report.json + out/episodes.jsonl
输出: web/data/
  report.json            运行级聚合 (原样)
  registry.json          指标注册表 (FR-5.1)
  index.json             回合索引(无大数组, fe-rq.md N2) + 图表预聚合(C1/C2/C3/C5)
  episodes/<id>.json     单回合全量 (调试页按需加载, O-F3)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pabench.schema import Episode
from pabench.webdata import build_index, index_record
from pabench.metrics.registry import METRIC_REGISTRY


def main():
    root = Path(__file__).resolve().parent
    out_dir, web_data = root / "out", root / "web" / "data"
    if not (out_dir / "episodes.jsonl").exists():
        sys.exit("缺 out/episodes.jsonl — 先运行: python demo.py")
    ep_dir = web_data / "episodes"
    shutil.rmtree(web_data, ignore_errors=True)
    ep_dir.mkdir(parents=True)

    shutil.copy(out_dir / "report.json", web_data / "report.json")
    report = json.loads((out_dir / "report.json").read_text())
    (web_data / "registry.json").write_text(
        json.dumps(METRIC_REGISTRY, ensure_ascii=False, indent=1))

    records = []
    with open(out_dir / "episodes.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ep = Episode.from_dict(d)
            records.append(index_record(ep))
            (ep_dir / f"{ep.episode_id}.json").write_text(json.dumps(d))

    (web_data / "index.json").write_text(
        json.dumps(build_index(report, records), ensure_ascii=False))
    print(f"web 数据就绪: {len(records)} 回合索引 + 单回合文件 → {web_data}")
    print(f"index.json {((web_data/'index.json').stat().st_size/1024):.0f} KB "
          f"(列表不含大数组, NFR-FE N2)")


if __name__ == "__main__":
    main()
