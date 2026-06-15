#!/usr/bin/env python
"""PA-Bench Console 服务模式入口 (M-FE2, fe-rq.md §13):

  python serve.py [--port 8000] [--host 127.0.0.1]
  → http://127.0.0.1:8000/  (同一进程: FastAPI /api/* + 静态前端 web/)

已有 out/ 产物会被自动导入为历史运行 run-000-legacy-out;
新运行经浏览器 /runs/new 向导发起, 产物落盘 runs/<run_id>/。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from pabench.platform import create_app

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = create_app(runs_dir=ROOT / "runs", web_dir=ROOT / "web",
                     legacy_out=ROOT / "out")
    print(f"PA-Bench Console (M-FE2 服务模式): http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
