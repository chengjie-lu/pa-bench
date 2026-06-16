"""Execution backend: reproducibility (NFR-1), telemetry structure (FR-2.3), oracle replay (FR-2.5), MuJoCo stub degradation."""
import numpy as np
import pytest

from pabench.models import PreciseVLA
from pabench.runners import CALIBRATED_ARM, WORN_ARM
from pabench.runners.mujoco_sim import MUJOCO_AVAILABLE, MujocoBackend
from pabench.schema import Phase


def test_same_seed_identical_hash(backend, base_scene):
    """NFR-1: replaying with the same benchmark_version + seed → identical content hash."""
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
    # 100 Hz uniform grid, command and telemetry same length (FR-2.3: zero alignment error)
    assert ep.robot.t[1] - ep.robot.t[0] == pytest.approx(0.01)
    assert ep.model.chunk.cmd_xyz.shape == (n, 3)
    assert ep.robot.ee_xyz_actual.shape == (n, 3)
    assert ep.robot.ft_wrench.shape == (n, 6)
    # phase segmentation covers the whole run with no gaps
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
    """FR-2.5: the perfect-perception trajectory must succeed on the nominal scene with calibrated hardware — the precondition for the attribution control."""
    ep = backend.run_oracle(base_scene, CALIBRATED_ARM, seed=9)
    assert ep.outcome.success
    assert ep.model.model_id == "oracle-replay"
    assert ep.model.chunk.entropy is None  # oracle has no uncertainty → the metric should record N/A


def test_mujoco_backend_degrades_gracefully():
    """Heavy-dependency isolation: when mujoco is missing, import does not crash and instantiation gives clear guidance;
    when installed this degradation path does not apply (real-backend behavior is in test_mujoco_backend.py)."""
    if MUJOCO_AVAILABLE:
        pytest.skip("mujoco is installed — run the real-backend tests instead")
    with pytest.raises(RuntimeError, match="mujoco is not installed"):
        MujocoBackend()