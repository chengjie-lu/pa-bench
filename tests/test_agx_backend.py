"""AGX Dynamics 后端: 确定性 + 硬件区分 + 与解析后端同口径 (agx 缺失/无许可时跳过)。

AGX 是商用许可产品, CI 与本机通常不可用 → 与 MuJoCo 后端同样用 skipif 跳过物理断言。
即使 agx 不可用, 也要保证: 接口可发现 (BACKEND_IDS 含 agx)、降级路径给出明确指引。
"""
import pytest

from pabench.runners import BACKEND_IDS, make_backend
from pabench.runners.agx_sim import AGX_AVAILABLE, AgxBackend


def test_agx_registered_in_factory():
    # 后端工厂始终能发现 agx (即插即用契约), 与是否安装无关。
    assert "agx" in BACKEND_IDS


def test_agx_graceful_degradation():
    if AGX_AVAILABLE:
        pytest.skip("agx 可用, 不测降级路径")
    with pytest.raises(RuntimeError, match="AGX Dynamics"):
        AgxBackend()


pytestmark_phys = pytest.mark.skipif(not AGX_AVAILABLE, reason="agx 未安装/无许可")

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
