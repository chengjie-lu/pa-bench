#!/usr/bin/env python
"""PA-Bench Console server-mode entry point (M-FE2, fe-rq.md §13):

  python serve.py [--port 8000] [--host 127.0.0.1]
  → http://127.0.0.1:8000/  (one process: FastAPI /api/* + static frontend web/)

Existing out/ artifacts are auto-imported as the historical run run-000-legacy-out;
new runs are launched via the browser /runs/new wizard, with artifacts persisted to runs/<run_id>/.
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
    print(f"PA-Bench Console (M-FE2 server mode): http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
