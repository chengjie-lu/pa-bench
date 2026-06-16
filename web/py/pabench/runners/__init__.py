from .base import Backend, HardwareProfile
from .fake_sim import CALIBRATED_ARM, WORN_ARM, FakeSimBackend

# Backend factory (NFR-5 plug-and-play): RunConfig.backend / the API pick a backend by name.
# Physics backends (mujoco / agx) are imported lazily — a missing library/license does not affect the default fake path.
BACKEND_IDS = ("fake", "mujoco", "agx")


def make_backend(backend_id: str = "fake", dt: float = None) -> Backend:
    """Instantiate an execution backend by id. With dt=None each backend uses its own default control step."""
    if backend_id == "fake":
        return FakeSimBackend() if dt is None else FakeSimBackend(dt)
    if backend_id == "mujoco":
        from .mujoco_sim import MujocoBackend
        return MujocoBackend() if dt is None else MujocoBackend(dt)
    if backend_id == "agx":
        from .agx_sim import AgxBackend
        return AgxBackend() if dt is None else AgxBackend(dt)
    raise ValueError(f"unknown backend: {backend_id!r} (choices: {BACKEND_IDS})")
