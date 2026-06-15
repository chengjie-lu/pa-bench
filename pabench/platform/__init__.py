"""M-FE2 服务化平台层 (fe-rq.md §8/§13, rq.md §9 预留目录)。

注: O-F4 建议的仓库顶层 platform/ 与 Python 标准库 platform 模块同名,
会遮蔽 stdlib (numpy 等内部 import platform 即炸), 故落在 pabench.platform 子包。
"""
from .run_manager import RunManager
from .api import create_app

__all__ = ["RunManager", "create_app"]
