"""执行后端: 复现性 (NFR-1)、遥测结构 (FR-2.3)、oracle 回放 (FR-2.5)、MuJoCo 桩降级。"""
import numpy as np
import pytest

from pabench.models import PreciseVLA
from pabench.runners import CALIBRATED_ARM, WORN_ARM
from pabench.runners.mujoco_sim import MUJOCO_AVAILABLE, MujocoBackend
from pabench.schema import Phase


def test_same_seed_identical_hash(backend, base_scene):
    """NFR-1: 同 benchmark_version + seed 重放 → 内容哈希一致。"""
    e1 = backend.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=123)
    e2 = backend.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=123)
    assert e1.content_hash() == e2.content_hash()


def test_different_seed_differs(backend, base_scene):
    e1 = backend.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=123)
    e2 = backend.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=124)
    assert e1.content_hash() != e2.content_hash()


def test_telemetry_structure(backend, base_scene):
    ep = backend.run_episode(base_scene, PreciseVLA(), CALIBRATED_ARM, seed=5)
    n = len(ep.robot.t)
    # 100 Hz 统一网格, 指令与遥测同长 (FR-2.3: 对齐误差 0)
    assert ep.robot.t[1] - ep.robot.t[0] == pytest.approx(0.01)
    assert ep.model.chunk.cmd_xyz.shape == (n, 3)
    assert ep.robot.ee_xyz_actual.shape == (n, 3)
    assert ep.robot.ft_wrench.shape == (n, 6)
    # phase 切分覆盖全程且无缝
    spans = ep.robot.phase_spans
    assert spans[0][1] == 0 and spans[-1][2] == n
    for (_, _, i1), (_, j0, _) in zip(spans, spans[1:]):
        assert i1 == j0
    assert [p for p, _, _ in spans] == [Phase.APPROACH, Phase.GRASP, Phase.TRANSFER,
                                        Phase.ALIGN, Phase.INSERT, Phase.FASTEN]


def test_failure_fields_consistent(sloppy_calibrated):
    for ep in sloppy_calibrated:
        if ep.outcome.success:
            assert ep.outcome.failure_phase is None
            assert ep.outcome.phase_reached is Phase.DONE
        else:
            assert ep.outcome.failure_phase in (Phase.GRASP, Phase.INSERT)
            assert ep.outcome.failure_label in ("grasp_miss", "insertion_misalign")


def test_oracle_succeeds_on_calibrated_nominal(backend, base_scene):
    """FR-2.5: 完美感知轨迹在校准硬件上的标称场景必须成功 — 归因对照的前提。"""
    ep = backend.run_oracle(base_scene, CALIBRATED_ARM, seed=9)
    assert ep.outcome.success
    assert ep.model.model_id == "oracle-replay"
    assert ep.model.chunk.entropy is None  # oracle 无不确定性 → 指标应记 N/A


def test_mujoco_backend_degrades_gracefully():
    """重型依赖隔离: mujoco 缺失时 import 不崩溃、实例化给出明确指引;
    已安装时此降级路径不适用 (真后端行为见 test_mujoco_backend.py)。"""
    if MUJOCO_AVAILABLE:
        pytest.skip("mujoco 已安装 — 走真后端测试")
    with pytest.raises(RuntimeError, match="mujoco 未安装"):
        MujocoBackend()