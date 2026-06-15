from .base import Backend, HardwareProfile
from .fake_sim import CALIBRATED_ARM, WORN_ARM, FakeSimBackend

# 后端工厂 (NFR-5 即插即用): RunConfig.backend / API 按名取后端。
# 物理后端 (mujoco / agx) 惰性导入 —— 缺库/缺许可不影响默认 fake 链路。
BACKEND_IDS = ("fake", "mujoco", "agx")


def make_backend(backend_id: str = "fake", dt: float = None) -> Backend:
    """按 id 实例化执行后端。dt=None 时各后端用自身默认控制步长。"""
    if backend_id == "fake":
        return FakeSimBackend() if dt is None else FakeSimBackend(dt)
    if backend_id == "mujoco":
        from .mujoco_sim import MujocoBackend
        return MujocoBackend() if dt is None else MujocoBackend(dt)
    if backend_id == "agx":
        from .agx_sim import AgxBackend
        return AgxBackend() if dt is None else AgxBackend(dt)
    raise ValueError(f"未知后端: {backend_id!r} (可选: {BACKEND_IDS})")
