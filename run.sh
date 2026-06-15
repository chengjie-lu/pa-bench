#!/usr/bin/env bash
# PA-Bench 一键运行: 装依赖 → 跑测试 → 跑端到端 demo → 列出产物
# 用法:
#   ./run.sh                  # 用 PATH 里的 python
#   PYTHON=/path/to/python ./run.sh   # 指定解释器 (如 anaconda env)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
echo "== 使用解释器: $($PYTHON -c 'import sys; print(sys.executable)') =="
$PYTHON --version

echo
echo "== [1/3] 安装依赖 =="
$PYTHON -m pip install -q -r requirements.txt
echo "依赖就绪: $($PYTHON -c 'import numpy, pytest, fastapi; print("numpy", numpy.__version__, "/ pytest", pytest.__version__, "/ fastapi", fastapi.__version__)')"

echo
echo "== [2/3] 测试套件 =="
$PYTHON -m pytest tests/ -q

echo
echo "== [3/3] 端到端 demo (2 模型 × 2 硬件 × 28 回合) =="
$PYTHON demo.py --episodes 24 --seed 7 --out out

echo
echo "== [4/4] 构建 Web 控制台数据 (M-FE1 静态模式) =="
$PYTHON build_web_data.py

echo
echo "== 产物 =="
ls -lh out/
echo
echo "静态报告:  open out/report.html"
echo "Web 控制台 (M-FE1 静态模式): $PYTHON -m http.server 8765 --directory web"
echo "           然后访问 http://localhost:8765/"
echo "Web 控制台 (M-FE2 服务模式): $PYTHON serve.py"
echo "           浏览器发起评测 + 实时进度, 访问 http://127.0.0.1:8000/"