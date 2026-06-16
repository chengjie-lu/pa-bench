"""M-FE2 service platform layer (fe-rq.md §8/§13, rq.md §9 reserved directory).

Note: the repo-top-level platform/ suggested by O-F4 has the same name as Python's stdlib platform module
and would shadow it (numpy and others do `import platform` internally and would break), so it lives in the pabench.platform subpackage.
"""
from .run_manager import RunManager
from .api import create_app

__all__ = ["RunManager", "create_app"]
