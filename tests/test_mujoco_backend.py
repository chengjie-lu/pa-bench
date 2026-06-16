"""MuJoCo physics backend: determinism + hardware discrimination + same conventions as the analytic backend (skipped when mujoco is missing)."""
import pytest

from pabench.runners.mujoco_sim import MUJOCO_AVAILABLE, MujocoBackend

pytestmark = pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="mujoco is not installed")

from pabench.models import PreciseVLA  # noqa: E402
from pabench.runners import CALIBRATED_ARM, WORN_ARM  # noqa: E402
from pabench.metrics import tracking_error  # noqa: E402


@pytest.fixture(scope="module")
def mj():
    return MujocoBackend()


def test_mj_determinism(mj, base_scene):
    e1 = mj.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=11)
    e2 = mj.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=11)
    assert e1.content_hash() == e2.content_hash()


def test_mj_worn_arm_tracks_worse(mj, base_scene):
    t_cal = tracking_error(mj.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=3))
    t_worn = tracking_error(mj.run_episode(base_scene, PreciseVLA(), WORN_ARM, seed=3))
    assert t_worn["steady_rms_m"] > 3 * t_cal["steady_rms_m"]


def test_mj_schema_compatible(mj, base_scene):
    ep = mj.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=5)
    n = len(ep.robot.t)
    assert ep.model.chunk.cmd_xyz.shape == (n, 3)
    assert ep.robot.ee_xyz_actual.shape == (n, 3)
    assert ep.robot.phase_spans[-1][2] == n
    assert ep.benchmark_version.endswith("+mujoco")
