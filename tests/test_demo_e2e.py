"""端到端: demo CLI 全链路跑通并产出三件套 (报告 HTML/JSON + episodes.jsonl)。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import demo  # noqa: E402


def test_demo_end_to_end(tmp_path):
    rc = demo.main(["--episodes", "8", "--seed", "3", "--out", str(tmp_path)])
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["results"]) == 4  # 2 模型 × 2 硬件
    for r in report["results"]:
        assert 0.0 <= r["sr"] <= 1.0
        assert r["ci95"][0] <= r["sr"] <= r["ci95"][1]
    html = (tmp_path / "report.html").read_text()
    assert "precise-vla-0.3" in html and "sloppy-vla-0.1" in html
    # episodes.jsonl 行数 = 4 组合 × (1 nominal + 8 mutation + 3 MR)
    lines = (tmp_path / "episodes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4 * (1 + 8 + 3)
    ep0 = json.loads(lines[0])
    assert ep0["benchmark_version"].startswith("pa-bench-")