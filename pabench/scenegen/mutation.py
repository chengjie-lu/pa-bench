"""FR-1.2 mutation-test generation — parametric perturbation of the nominal scene, with all parameters written to perturbation. Real implementation.

Perturbation dimensions: part pose (dx/dy/dyaw), lighting lux_factor (affects model perception noise), friction.
friction is recorded per the schema but this slice's FakeSimBackend does not consume it (it takes effect with the MuJoCo backend).
"""
from __future__ import annotations

import numpy as np

from ..schema import GenerationMethod, SE3Pose, Scene


class MutationGenerator:
    def __init__(self, seed: int,
                 pos_range_m: float = 0.015,
                 yaw_range_rad: float = 0.3,
                 lux_range=(0.3, 1.0),
                 friction_range=(0.6, 1.2)):
        self.rng = np.random.default_rng(seed)
        self.pos_range_m = pos_range_m
        self.yaw_range_rad = yaw_range_rad
        self.lux_range = lux_range
        self.friction_range = friction_range

    def generate(self, base: Scene, parent_episode_id: str, n: int) -> list[Scene]:
        scenes = []
        for i in range(n):
            dx = float(self.rng.uniform(-self.pos_range_m, self.pos_range_m))
            dy = float(self.rng.uniform(-self.pos_range_m, self.pos_range_m))
            dyaw = float(self.rng.uniform(-self.yaw_range_rad, self.yaw_range_rad))
            lux = float(self.rng.uniform(*self.lux_range))
            friction = float(self.rng.uniform(*self.friction_range))
            p = base.part_pose_gt
            scenes.append(Scene(
                scene_id=f"{base.scene_id}-mut{i:03d}",
                task_type=base.task_type,
                tolerance_class=base.tolerance_class,
                part_pose_gt=SE3Pose((p.xyz[0] + dx, p.xyz[1] + dy, p.xyz[2]), p.yaw + dyaw),
                target_pose_gt=base.target_pose_gt,
                perturbation={"part_dx": dx, "part_dy": dy, "part_dyaw": dyaw,
                              "lux_factor": lux, "friction": friction},
                generation_method=GenerationMethod.MUTATION,
                parent_episode_id=parent_episode_id,
            ))
        return scenes