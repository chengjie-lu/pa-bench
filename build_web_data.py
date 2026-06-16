#!/usr/bin/env python
"""M-FE1 static-adaptation layer (fe-rq.md §13): pre-split/pre-aggregate out/ artifacts into static data the frontend can fetch directly.

The build logic lives in pabench/webdata.py (from M-FE2 on, shared with the platform API).

Input: out/report.json + out/episodes.jsonl
Output: web/data/
  report.json            run-level aggregation (as-is)
  registry.json          metric registry (FR-5.1)
  index.json             episode index (no large arrays, fe-rq.md N2) + chart pre-aggregations (C1/C2/C3/C5)
  episodes/<id>.json     full per-episode payload (loaded on demand by the debug page, O-F3)

Also emits web/py/ — the pabench Python sources + a manifest — so the Pyodide in-browser
runtime can run evaluations and compute custom metrics client-side (no backend) on the static
deploy. The FastAPI platform/ layer is excluded (not importable / not needed in the browser).
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


def export_py_sources(root: Path, web_dir: Path):
    """Copy the pabench package sources (minus the FastAPI platform/ layer) + a manifest into
    web/py/, so the Pyodide worker can load and import them in the browser."""
    py_dir = web_dir / "py"
    pkg_dst = py_dir / "pabench"
    # only clear the generated package copy — keep the hand-written worker.js / pyboot.js
    shutil.rmtree(pkg_dst, ignore_errors=True)
    py_dir.mkdir(parents=True, exist_ok=True)
    pkg_src = root / "pabench"
    manifest = []
    for p in sorted(pkg_src.rglob("*.py")):
        rel = p.relative_to(pkg_src)
        if rel.parts[0] in ("platform", "__pycache__"):
            continue  # FastAPI layer not needed/importable in the browser
        dst = pkg_dst / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)
        manifest.append(f"pabench/{rel.as_posix()}")
    (py_dir / "manifest.json").write_text(json.dumps({"files": manifest}, indent=1))
    return manifest


def main():
    root = Path(__file__).resolve().parent
    out_dir, web_data = root / "out", root / "web" / "data"
    if not (out_dir / "episodes.jsonl").exists():
        sys.exit("missing out/episodes.jsonl — run first: python demo.py")
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
    manifest = export_py_sources(root, root / "web")
    print(f"web data ready: {len(records)} episode-index rows + per-episode files → {web_data}")
    print(f"index.json {((web_data/'index.json').stat().st_size/1024):.0f} KB "
          f"(list carries no large arrays, NFR-FE N2)")
    print(f"web/py ready: {len(manifest)} pabench source files for the Pyodide runtime")


if __name__ == "__main__":
    main()
