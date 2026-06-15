"""场景生成: 变异确定性与参数记录 (FR-1.2), MR-1 变质关系 (FR-1.3)。"""
import math

import numpy as np
import pytest

from pabench.models import PreciseVLA, SloppyVLA
from pabench.runners import CALIBRATED_ARM
from pabench.scenegen import (MR1RotationZ, MutationGenerator, dtw_mean_distance,
                              mr_violation_verdict)
from pabench.schema import GenerationMethod


def test_dtw_mean_distance_basics():
    a = np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)])
    assert dtw_mean_distance(a, a) == pytest.approx(0.0)
    b = a.copy()
    b[:, 1] += 0.002  # 垂直于路径的恒定偏置无法被规整吸收
    assert dtw_mean_distance(a, b) == pytest.approx(0.002, rel=0.05)


def test_mutation_deterministic_and_recorded(base_scene):
    a = MutationGenerator(seed=42).generate(base_scene, "anchor", 10)
    b = MutationGenerator(seed=42).generate(base_scene, "anchor", 10)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    for s in a:
        assert s.generation_method is GenerationMethod.MUTATION
        assert s.parent_episode_id == "anchor"
        assert set(s.perturbation) == {"part_dx", "part_dy", "part_dyaw", "lux_factor", "friction"}
        assert 0.3 <= s.perturbation["lux_factor"] <= 1.0
        assert abs(s.perturbation["part_dx"]) <= 0.015
        # 扰动确实作用在场景真值上
        assert s.part_pose_gt.xyz[0] == pytest.approx(
            base_scene.part_pose_gt.xyz[0] + s.perturbation["part_dx"])


def test_mutation_different_seed_differs(base_scene):
    a = MutationGenerator(seed=1).generate(base_scene, "anchor", 5)
    b = MutationGenerator(seed=2).generate(base_scene, "anchor", 5)
    assert [s.to_dict() for s in a] != [s.to_dict() for s in b]


def test_mr1_scene_geometry(base_scene):
    mr = MR1RotationZ(math.pi / 2)
    rotated = mr.apply(base_scene, "parent-ep")
    x, y, z = base_scene.target_pose_gt.xyz
    assert rotated.target_pose_gt.xyz[0] == pytest.approx(-y)
    assert rotated.target_pose_gt.xyz[1] == pytest.approx(x)
    assert rotated.target_pose_gt.xyz[2] == pytest.approx(z)
    assert rotated.mr_id == "MR-1"
    assert rotated.parent_episode_id == "parent-ep"
    assert rotated.generation_method is GenerationMethod.METAMORPHIC


def test_mr1_precise_passes_sloppy_violates(backend, base_scene):
    """等变模型应通过 MR-1; 带世界系偏置的模型应违反 — 这是归因 rule1 的依据。
    按协议判定: 多旋转后继的中位距离 (单次比较会被感知噪声左右)。"""
    thetas = [math.pi / 2, 2 * math.pi / 3, 5 * math.pi / 6]
    for model, expect_violated in [(PreciseVLA(), False), (SloppyVLA(), True)]:
        src = backend.run_episode(base_scene, model, CALIBRATED_ARM, seed=300)
        checks = []
        for j, theta in enumerate(thetas):
            mr = MR1RotationZ(theta)
            fol = backend.run_episode(mr.apply(base_scene, src.episode_id),
                                      model, CALIBRATED_ARM, seed=301 + j)
            checks.append(mr.check(src, fol))
        verdict = mr_violation_verdict(checks)
        assert verdict["violated"] is expect_violated, (
            f"{model.model_id}: median={verdict['median_dist_m']*1e3:.2f}mm, "
            f"threshold={verdict['threshold_m']*1e3:.2f}mm")
        assert verdict["n_followups"] == 3