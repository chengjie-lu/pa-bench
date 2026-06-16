#!/usr/bin/env bash
# PA-Bench one-shot run: install deps → run tests → run end-to-end demo → list artifacts
# Usage:
#   ./run.sh                  # use the python on PATH
#   PYTHON=/path/to/python ./run.sh   # pick an interpreter (e.g. an anaconda env)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
echo "== using interpreter: $($PYTHON -c 'import sys; print(sys.executable)') =="
$PYTHON --version

echo
echo "== [1/3] install dependencies =="
$PYTHON -m pip install -q -r requirements.txt
echo "deps ready: $($PYTHON -c 'import numpy, pytest, fastapi; print("numpy", numpy.__version__, "/ pytest", pytest.__version__, "/ fastapi", fastapi.__version__)')"

echo
echo "== [2/3] test suite =="
$PYTHON -m pytest tests/ -q

echo
echo "== [3/3] end-to-end demo (2 models × 2 hardware × 28 episodes) =="
$PYTHON demo.py --episodes 24 --seed 7 --out out

echo
echo "== [4/4] build web console data (M-FE1 static mode) =="
$PYTHON build_web_data.py

echo
echo "== artifacts =="
ls -lh out/
echo
echo "static report:  open out/report.html"
echo "web console (M-FE1 static mode): $PYTHON -m http.server 8765 --directory web"
echo "           then open http://localhost:8765/"
echo "web console (M-FE2 server mode): $PYTHON serve.py"
echo "           launch runs from the browser + live progress, open http://127.0.0.1:8000/"