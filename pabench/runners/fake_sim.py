"""[STUB / FAKE backend] analytic-kinematics fake simulation — substitute for MuJoCo (not installed here).

It is a complete implementation of the Backend interface; physics is approximated with a first-order
tracking filter + systematic drift + sinusoidal jitter + white noise, which is enough to produce all
the signals the evaluation pipeline needs (command/actual trajectory separation, force interaction,
success criterion). When wiring up MuJoCo, implement runners/mujoco_sim.MujocoBackend; the upper layers
need zero changes.

Error chain (supports the FR-4.2 e_plan/e_track decomposition):
  final assembly error = model perception error (e_plan, in the command) + hardware tracking error (e_track, actual − command)
"""
from __future__ import annotations

import numpy as np

from ..schema import (BENCHMARK_VERSION, ActionChunk, Episode, ModelTrace, Outcome,
                      Phase, RobotTrace, Scene, Source)
from ..scenegen.nominal import PHASE_PLAN
from ..models.base import Observation, VLAModel
from ..models.fake import _HOME, _min_jerk
from .base import Backend, HardwareProfile

CALIBRATED_ARM = HardwareProfile(
    hw_config_id="arm-calibrated-2026Q2",
    tracking_alpha=0.35, droop_xy=(0.05e-3, -0.03e-3),
    track_noise_std=0.05e-3, jitter_amp=0.04e-3)

WORN_ARM = HardwareProfile(
    hw_config_id="arm-worn-2023Q1",
    tracking_alpha=0.12, droop_xy=(0.70e-3, -0.40e-3),
    track_noise_std=0.35e-3, jitter_amp=0.45e-3)

DT = 0.01                 # 100 Hz (rq.md §5: telemetry ≥100 Hz)
GRASP_TOL_M = 6.0e-3      # grasp-phase success criterion: commanded grasp point < 6 mm from part ground truth
FORCE_K = 5000.0          # insertion contact stiffness [N/m] (lateral deviation beyond tolerance → contact force)


def _phase_spans(dt: float):
    spans, i0 = [], 0
    for phase, dur in PHASE_PLAN:
        n = int(round(dur / dt))
        spans.append((phase, i0, i0 + n))
        i0 += n
    return spans, i0


class FakeSimBackend(Backend):
    def __init__(self, dt: float = DT):
        self.dt = dt
        self.phase_spans, self.n_steps = _phase_spans(dt)
        self.t_grid = np.arange(self.n_steps) * dt

    # ------------------------------------------------------------ public

    def run_episode(self, scene: Scene, model: VLAModel,
                    hw: HardwareProfile, seed: int) -> Episode:
        rng_model = np.random.default_rng(seed)             # model perception/inference noise
        obs = Observation(scene=scene,
                          lux_factor=float(scene.perturbation.get("lux_factor", 1.0)),
                          instruction="pick the cap from the bin and screw it onto the bottle",
                          t_grid=self.t_grid, phase_spans=self.phase_spans)
        chunk = model.infer(obs, rng_model)
        return self._execute(scene, model.model_id, chunk, hw, seed)

    def run_oracle(self, scene: Scene, hw: HardwareProfile, seed: int) -> Episode:
        """FR-2.5: perfect-perception (ground-truth waypoints) min-jerk expert trajectory, executed on the same hardware."""
        chunk = self._plan_oracle(scene)
        return self._execute(scene, "oracle-replay", chunk, hw, seed)

    # ------------------------------------------------------------ internals

    def _plan_oracle(self, scene: Scene) -> ActionChunk:
        part = np.array(scene.part_pose_gt.xyz)
        target = np.array(scene.target_pose_gt.xyz)
        wp_end = {
            Phase.APPROACH: part + [0, 0, 0.10],
            Phase.GRASP: part + [0, 0, 0.005],
            Phase.TRANSFER: target + [0, 0, 0.06],
            Phase.ALIGN: target + [0, 0, 0.015],
            Phase.INSERT: target + [0, 0, 0.004],
            Phase.FASTEN: target + [0, 0, 0.004],
        }
        cmd = np.zeros((self.n_steps, 3))
        yaw = np.zeros(self.n_steps)
        prev = _HOME
        for phase, i0, i1 in self.phase_spans:
            cmd[i0:i1] = _min_jerk(prev, wp_end[phase], i1 - i0)
            if phase is Phase.FASTEN:
                yaw[i0:i1] = np.linspace(0.0, 2 * np.pi, i1 - i0)
            prev = wp_end[phase]
        return ActionChunk(t=self.t_grid.copy(), cmd_xyz=cmd, cmd_yaw=yaw,
                           entropy=None, latency_ms=np.full(self.n_steps, 1.0))

    def _execute(self, scene: Scene, model_id: str, chunk: ActionChunk,
                 hw: HardwareProfile, seed: int) -> Episode:
        rng_hw = np.random.default_rng(seed + 10_000_019)   # independent hardware-noise stream
        n = self.n_steps
        cmd = chunk.cmd_xyz

        # 1) first-order tracking filter (execution lag)
        actual = np.zeros_like(cmd)
        state = cmd[0].copy()
        a = hw.tracking_alpha
        for i in range(n):
            state = state + a * (cmd[i] - state)
            actual[i] = state
        # 2) systematic drift + jitter (5–50 Hz sinusoids) + white noise
        actual[:, 0] += hw.droop_xy[0]
        actual[:, 1] += hw.droop_xy[1]
        phases = rng_hw.uniform(0, 2 * np.pi, (len(hw.jitter_freqs), 3))
        for k, f in enumerate(hw.jitter_freqs):
            for ax in range(3):
                actual[:, ax] += hw.jitter_amp * np.sin(2 * np.pi * f * self.t_grid + phases[k, ax])
        actual += rng_hw.normal(0.0, hw.track_noise_std, actual.shape)

        yaw_actual = chunk.cmd_yaw.copy()  # yaw tracking is approximately ideal (vertical-slice simplification)

        # 3) success criterion (FR-3.1 machine-checkable function)
        gap = scene.tolerance_class.gap_m
        part_xy = np.array(scene.part_pose_gt.xyz[:2])
        target_xy = np.array(scene.target_pose_gt.xyz[:2])
        _, g0, g1 = next(s for s in self.phase_spans if s[0] is Phase.GRASP)
        _, _, ins_end = next(s for s in self.phase_spans if s[0] is Phase.INSERT)
        grasp_err = float(np.linalg.norm(cmd[g1 - 1, :2] - part_xy))
        align_err = float(np.linalg.norm(actual[ins_end - 1, :2] - target_xy))

        if grasp_err >= GRASP_TOL_M:
            success, phase_reached, failure_phase = False, Phase.APPROACH, Phase.GRASP
            failure_label = "grasp_miss"
        elif align_err >= gap:
            success, phase_reached, failure_phase = False, Phase.ALIGN, Phase.INSERT
            failure_label = "insertion_misalign"
        else:
            success, phase_reached, failure_phase, failure_label = True, Phase.DONE, None, None

        # 4) contact force (insert + fasten phases; lateral deviation beyond tolerance produces contact force → FR-3.7)
        wrench = np.zeros((n, 6))
        for ph in (Phase.INSERT, Phase.FASTEN):
            _, i0, i1 = next(s for s in self.phase_spans if s[0] is ph)
            lat = actual[i0:i1, :2] - target_xy[None, :]
            lat_n = np.linalg.norm(lat, axis=1)
            over = np.clip(lat_n - gap, 0.0, None)
            with np.errstate(invalid="ignore", divide="ignore"):
                unit = np.where(lat_n[:, None] > 0, lat / np.maximum(lat_n[:, None], 1e-12), 0.0)
            wrench[i0:i1, 0:2] = FORCE_K * over[:, None] * unit

        # 5) gripper
        gripper = np.full(n, 0.04)
        gripper[g1:] = 0.012

        robot = RobotTrace(t=self.t_grid.copy(), ee_xyz_actual=actual,
                           ee_yaw_actual=yaw_actual, ft_wrench=wrench,
                           gripper_width=gripper, hw_config_id=hw.hw_config_id,
                           phase_spans=list(self.phase_spans))
        outcome = Outcome(success=success, phase_reached=phase_reached,
                          failure_phase=failure_phase, failure_label=failure_label,
                          attribution=None, duration_s=float(self.t_grid[-1] + self.dt))
        return Episode(
            episode_id=f"{scene.scene_id}__{model_id}__{hw.hw_config_id}__s{seed}",
            benchmark_version=BENCHMARK_VERSION, source=Source.SIM, scene=scene,
            model=ModelTrace(model_id=model_id,
                             language_instruction="pick the cap from the bin and screw it onto the bottle",
                             chunk=chunk),
            robot=robot, outcome=outcome,
            media={"video_uris": [], "sim_state_log_uri": None},
        )