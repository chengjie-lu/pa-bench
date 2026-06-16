"""AGX Dynamics backend: determinism + hardware discrimination + same conventions as the analytic backend (skipped when agx is missing / unlicensed).

AGX is a commercially licensed product; CI and local machines usually can't use it → like the MuJoCo backend,
its physics assertions are skipped via skipif. Even when agx is unavailable we still guarantee: the interface
is discoverable (BACKEND_IDS contains agx) and the degradation path gives clear guidance.
"""
import pytest

from pabench.runners import BACKEND_IDS, make_backend
from pabench.runners.agx_sim import AGX_AVAILABLE, AgxBackend


def test_agx_registered_in_factory():
    # The backend factory always discovers agx (plug-and-play contract), regardless of whether it is installed.
    assert "agx" in BACKEND_IDS


def test_agx_graceful_degradation():
    if AGX_AVAILABLE:
        pytest.skip("agx is available, not testing the degradation path")
    with pytest.raises(RuntimeError, match="AGX Dynamics"):
        AgxBackend()


pytestmark_phys = pytest.mark.skipif(not AGX_AVAILABLE, reason="agx is not installed / unlicensed")

from pabench.models import PreciseVLA  # noqa: E402
from pabench.runners import CALIBRATED_ARM, WORN_ARM  # noqa: E402
from pabench.metrics import tracking_error  # noqa: E402


@pytest.fixture(scope="module")
def agx():
    return AgxBackend()


@pytestmark_phys
def test_agx_factory_returns_backend():
    assert isinstance(make_backend("agx"), AgxBackend)


@pytestmark_phys
def test_agx_determinism(agx, base_scene):
    e1 = agx.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=11)
    e2 = agx.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=11)
    assert e1.content_hash() == e2.content_hash()


@pytestmark_phys
def test_agx_worn_arm_tracks_worse(agx, base_scene):
    t_cal = tracking_error(agx.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=3))
    t_worn = tracking_error(agx.run_episode(base_scene, PreciseVLA(), WORN_ARM, seed=3))
    assert t_worn["steady_rms_m"] > 3 * t_cal["steady_rms_m"]


@pytestmark_phys
def test_agx_schema_compatible(agx, base_scene):
    ep = agx.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=5)
    n = len(ep.robot.t)
    assert ep.model.chunk.cmd_xyz.shape == (n, 3)
    assert ep.robot.ee_xyz_actual.shape == (n, 3)
    assert ep.robot.phase_spans[-1][2] == n
    assert ep.benchmark_version.endswith("+agx")