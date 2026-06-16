"""[STUB / FAKE] two scripted fake VLA models — substitutes for real model weights (which this environment can't host/run).

They implement the VLAModel interface, with controllable, mutually distinguishable behavior, to drive the whole evaluation chain:
- PreciseVLA: small perception noise, min-jerk smooth trajectory, frame-equivariant (MR-1 should pass),
  uncertainty output strongly correlated with its true perception error (AUROC should be high).
- SloppyVLA : a world-frame fixed perception bias (→ non-equivariant, MR-1 should be violated) + large noise + step-wise wobble
  (jerk should be high), uncertainty output carries no information.

To wire up a real model: create a VLAModel subclass that calls a remote inference service and delete/bypass this file.
"""
from __future__ import annotations

import math

import numpy as np

from ..schema import ActionChunk, Phase, SE3Pose
from .base import Observation, VLAModel

_HOME = np.array([0.30, 0.0, 0.30])


def _min_jerk(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, n)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return p0[None, :] + s[:, None] * (p1 - p0)[None, :]


def _linear(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return p0[None, :] + s[:, None] * (p1 - p0)[None, :]


class _ScriptedVLA(VLAModel):
    """Shared trajectory script: perceive (with each model's error model) → generate waypoints per phase → interpolate into a command trajectory."""

    sigma_base_m: float = 0.0          # perception noise (scaled by 1/lux, simulating low-light degradation)
    bias_world_m = np.zeros(2)         # world-frame fixed bias (source of non-equivariance)
    wobble_m: float = 0.0              # step-wise command wobble (source of jerk)
    smooth = True

    def _perceive(self, pose: SE3Pose, lux: float, rng) -> np.ndarray:
        sigma = self.sigma_base_m / max(lux, 1e-6)
        err = self.bias_world_m + rng.normal(0.0, sigma, 2)
        return np.array([pose.xyz[0] + err[0], pose.xyz[1] + err[1], pose.xyz[2]]), err

    def _entropy(self, n: int, target_err: np.ndarray, rng) -> np.ndarray:
        raise NotImplementedError  # each subclass's own uncertainty head (allowed in the stub)

    def _latency(self, n: int, rng) -> np.ndarray:
        raise NotImplementedError

    def infer(self, obs: Observation, rng: np.random.Generator) -> ActionChunk:
        part_hat, _part_err = self._perceive(obs.scene.part_pose_gt, obs.lux_factor, rng)
        target_hat, target_err = self._perceive(obs.scene.target_pose_gt, obs.lux_factor, rng)

        # phase-end waypoints (all based on the "perceived" poses → naturally equivariant under scene rotation, unless there is a world-frame bias)
        wp_end = {
            Phase.APPROACH: part_hat + [0, 0, 0.10],
            Phase.GRASP: part_hat + [0, 0, 0.005],
            Phase.TRANSFER: target_hat + [0, 0, 0.06],
            Phase.ALIGN: target_hat + [0, 0, 0.015],
            Phase.INSERT: target_hat + [0, 0, 0.004],
            Phase.FASTEN: target_hat + [0, 0, 0.004],
        }
        n_total = len(obs.t_grid)
        cmd = np.zeros((n_total, 3))
        yaw = np.zeros(n_total)
        prev = _HOME
        interp = _min_jerk if self.smooth else _linear
        for phase, i0, i1 in obs.phase_spans:
            seg = interp(prev, wp_end[phase], i1 - i0)
            cmd[i0:i1] = seg
            if phase is Phase.FASTEN:  # fasten: yaw rotates 2π
                yaw[i0:i1] = np.linspace(0.0, 2 * math.pi, i1 - i0)
            prev = wp_end[phase]
        if self.wobble_m > 0:
            cmd = cmd + rng.normal(0.0, self.wobble_m, cmd.shape)
        return ActionChunk(
            t=obs.t_grid.copy(), cmd_xyz=cmd, cmd_yaw=yaw,
            entropy=self._entropy(n_total, target_err, rng),
            latency_ms=self._latency(n_total, rng),
        )


class PreciseVLA(_ScriptedVLA):
    model_id = "precise-vla-0.3"
    sigma_base_m = 0.35e-3
    bias_world_m = np.zeros(2)
    wobble_m = 0.0
    smooth = True

    def _entropy(self, n, target_err, rng):
        # well-calibrated: entropy ∝ this episode's true perception error (simulating ensemble variance's sensitivity to error)
        return 0.3 + 3000.0 * float(np.linalg.norm(target_err)) + 0.05 * rng.standard_normal(n)

    def _latency(self, n, rng):
        return np.clip(rng.normal(12.0, 2.0, n), 1.0, None)


class SloppyVLA(_ScriptedVLA):
    model_id = "sloppy-vla-0.1"
    sigma_base_m = 1.0e-3
    bias_world_m = 1.2e-3 * np.array([math.cos(0.7), math.sin(0.7)])  # world-frame bias → MR-1 violation
    wobble_m = 0.3e-3
    smooth = False

    def _entropy(self, n, target_err, rng):
        # uninformative: independent of the true error
        return 1.0 + 0.1 * rng.standard_normal(n)

    def _latency(self, n, rng):
        return np.clip(rng.normal(35.0, 8.0, n), 1.0, None)