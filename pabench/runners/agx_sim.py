"""AGX Dynamics physics backend — a real implementation of the Backend interface (rq.md O-1, NFR-5 plug-and-play).

Why add another physics backend:
- MuJoCo leans toward research / rapid prototyping; AGX Dynamics (Algoryx) is an industrial,
  commercially licensed physics engine that is closer to a real production line in constraint
  solving (a direct solver, not a spring approximation), contact fidelity, and integration with
  digital-twin toolchains — the high-fidelity tier for precision-assembly evaluation.
- All three backends (Fake / MuJoCo / AGX) implement the same `Backend` interface, so the upper
  pipeline / API / report can switch between them with zero changes (see RunConfig.backend, BACKEND_REGISTRY).

Physics modeling (same conventions as the MuJoCo backend, to keep cross-backend ranking comparable —
the crossbackend_eval idea):
- End-effector = a Cartesian gantry: world →[prismatic X]→ bx →[prismatic Y]→ by
  →[prismatic Z]→ bz →[hinge yaw]→ tool, 4-DOF total, matching MuJoCo's slide×3 + hinge.
- Position servo = a LockController on each constraint, compliance = 1/kp, damping = kv;
  the tracking lag/overshoot produced by the AGX direct solver is real constraint dynamics, not an analytic filter.
- Hardware profile (HardwareProfile.hw_config_id) → servo stiffness/damping + injected disturbance forces
  (constant force = drift/droop, sinusoidal force = jitter, Gaussian force = noise), applied on the tool body's xy.
- Contact dynamics of grasping/fastening are out of scope here (success criterion, wrench, gripper match
  FakeSim/MuJoCo); contact-level fidelity (AGX's strength) is future work.

On import/license unavailability it degrades gracefully: AGX_AVAILABLE=False, instantiation gives clear guidance.
AGX is a commercially licensed product (needs a license server / offline license file); CI and local machines
usually lack it → the corresponding tests pytest.mark.skipif skip, consistent with the MuJoCo backend.
"""
from __future__ import annotations

import numpy as np

from ..schema import (BENCHMARK_VERSION, Episode, ModelTrace, Outcome, Phase,
                      RobotTrace, Scene, Source)
from ..models.base import Observation, VLAModel
from .base import Backend, HardwareProfile
from .fake_sim import DT, FORCE_K, GRASP_TOL_M, _phase_spans

try:
    import agx
    import agxSDK
    import agxCollide
    AGX_AVAILABLE = True
except ImportError:
    AGX_AVAILABLE = False

_HOME = np.array([0.30, 0.0, 0.30])

# Hardware calibration profile → AGX servo/disturbance params.
# kp [N/m] higher = stiffer (compliance=1/kp); kv [N·s/m] joint damping; droop/jitter/noise units N.
# Values are aligned with runners/mujoco_sim.MJ_PROFILES so the two physics backends' SR rankings compare directly.
AGX_PROFILES = {
    "arm-calibrated-2026Q2": dict(kp=2000.0, kv=80.0, droop=(0.10, -0.06),
                                  jitter_f=0.30, noise_f=0.15),
    "arm-worn-2023Q1": dict(kp=300.0, kv=60.0, droop=(0.25, -0.14),
                            jitter_f=2.80, noise_f=1.00),
    "arm-pristine": dict(kp=3000.0, kv=100.0, droop=(0.04, -0.02),
                         jitter_f=0.12, noise_f=0.08),
}
JITTER_FREQS = (12.0, 27.0)
PHYS_DT = 0.002  # physics step; 100 Hz control → 5 physics substeps per control step (same as MuJoCo)


class AgxBackend(Backend):
    def __init__(self, dt: float = DT):
        if not AGX_AVAILABLE:
            raise RuntimeError(
                "agx (AGX Dynamics) is not installed or its license is unavailable.\n"
                "  Install: see the Python bindings (agxpy) in the Algoryx AGX Dynamics distribution;\n"
                "  License: after setting AGX_LICENSE / a license file, importing `agx` should succeed;\n"
                "  In the current environment use FakeSimBackend or MujocoBackend (same interface).")
        # The AGX global runtime only needs to be initialized once (idempotent).
        if not agx.isInitialized():
            agx.init()
        self.dt = dt
        self.substeps = int(round(dt / PHYS_DT))
        self.phase_spans, self.n_steps = _phase_spans(dt)
        self.t_grid = np.arange(self.n_steps) * dt

    # ------------------------------------------------------------ public

    def run_episode(self, scene: Scene, model: VLAModel,
                    hw: HardwareProfile, seed: int) -> Episode:
        rng_model = np.random.default_rng(seed)
        obs = Observation(scene=scene,
                          lux_factor=float(scene.perturbation.get("lux_factor", 1.0)),
                          instruction="pick the cap from the bin and screw it onto the bottle",
                          t_grid=self.t_grid, phase_spans=self.phase_spans)
        chunk = model.infer(obs, rng_model)
        return self._execute(scene, model.model_id, chunk, hw, seed)

    def run_oracle(self, scene: Scene, hw: HardwareProfile, seed: int) -> Episode:
        from .fake_sim import FakeSimBackend
        chunk = FakeSimBackend(self.dt)._plan_oracle(scene)  # same expert trajectory planner
        return self._execute(scene, "oracle-replay", chunk, hw, seed)

    # ------------------------------------------------------------ internals

    def _build_world(self, prof: dict):
        """Build the 4-DOF gantry + servos; return (sim, tool_body, [lock_x, lock_y, lock_z, lock_yaw]).

        A fresh independent Simulation is built per episode to guarantee determinism (no cross-episode state leak).
        """
        sim = agxSDK.Simulation()
        sim.setTimeStep(PHYS_DT)
        sim.setUniformGravity(agx.Vec3(0.0, 0.0, 0.0))  # same as MuJoCo gravity=0, to isolate gravity effects

        def _slider(mass: float, pos: np.ndarray) -> "agx.RigidBody":
            rb = agx.RigidBody()
            rb.add(agxCollide.Geometry(agxCollide.Sphere(0.01)))
            rb.getMassProperties().setMass(mass)
            rb.setPosition(agx.Vec3(float(pos[0]), float(pos[1]), float(pos[2])))
            sim.add(rb)
            return rb

        # Gantry chain: each stage carries the downstream ones; the final tool has mass 1 kg (matching the MuJoCo tool body).
        bx = _slider(1.0, _HOME)
        by = _slider(1.0, _HOME)
        bz = _slider(1.0, _HOME)
        tool = _slider(1.0, _HOME)

        kp, kv = prof["kp"], prof["kv"]
        compliance = 1.0 / kp

        def _prismatic(axis, rb1, rb2):
            # rb1 translates along world axis relative to rb2; rb2=None → relative to the world.
            pris = agx.Prismatic(agx.Vec3(*axis), rb1, rb2) if rb2 is not None \
                else agx.Prismatic(agx.Vec3(*axis), rb1)
            pris.getMotor1D().setEnable(False)
            lock = pris.getLock1D()
            lock.setEnable(True)
            lock.setCompliance(compliance)   # compliance = 1/kp → stiffness kp
            lock.setDamping(kv * PHYS_DT)     # AGX damping is measured as (damping coefficient · dt)
            sim.add(pris)
            return lock

        lock_x = _prismatic((1, 0, 0), bx, None)
        lock_y = _prismatic((0, 1, 0), by, bx)
        lock_z = _prismatic((0, 0, 1), bz, by)

        # yaw: tool rotates about world Z relative to bz
        hinge = agx.Hinge(agx.Vec3(0, 0, 1), tool, bz)
        hinge.getMotor1D().setEnable(False)
        lock_yaw = hinge.getLock1D()
        lock_yaw.setEnable(True)
        lock_yaw.setCompliance(1.0 / 50.0)   # yaw servo stiffness (aligned with MuJoCo kp=50)
        lock_yaw.setDamping(2.0 * PHYS_DT)
        sim.add(hinge)

        return sim, tool, (lock_x, lock_y, lock_z, lock_yaw)

    def _execute(self, scene: Scene, model_id: str, chunk, hw, seed: int) -> Episode:
        prof = AGX_PROFILES[hw.hw_config_id]
        sim, tool, (lx, ly, lz, lyaw) = self._build_world(prof)
        rng_hw = np.random.default_rng(seed + 10_000_019)
        phases = rng_hw.uniform(0, 2 * np.pi, (len(JITTER_FREQS), 2))
        n = self.n_steps
        cmd = chunk.cmd_xyz
        actual = np.zeros((n, 3))
        yaw_actual = np.zeros(n)
        droop = np.array(prof["droop"])

        for i in range(n):
            # position-servo target (joint frame = displacement relative to HOME)
            tgt = cmd[i] - _HOME
            lx.setPosition(float(tgt[0]))
            ly.setPosition(float(tgt[1]))
            lz.setPosition(float(tgt[2]))
            lyaw.setPosition(float(chunk.cmd_yaw[i]))
            # disturbance force: drift + jitter + noise (applied to the tool body xy)
            f = droop.copy()
            for k, fr in enumerate(JITTER_FREQS):
                f += prof["jitter_f"] * np.sin(2 * np.pi * fr * self.t_grid[i] + phases[k])
            f += rng_hw.normal(0.0, prof["noise_f"], 2)
            for _ in range(self.substeps):
                tool.setForce(agx.Vec3(float(f[0]), float(f[1]), 0.0))
                sim.stepForward()
            p = tool.getPosition()
            actual[i] = (p.x(), p.y(), p.z())
            # tool rotation about Z (quaternion → yaw)
            rot = tool.getRotation()
            yaw_actual[i] = float(np.arctan2(
                2.0 * (rot.w() * rot.z() + rot.x() * rot.y()),
                1.0 - 2.0 * (rot.y() ** 2 + rot.z() ** 2)))

        # success criterion and wrench/gripper — same as FakeSim / MuJoCo (cross-backend comparable)
        gap = scene.tolerance_class.gap_m
        part_xy = np.array(scene.part_pose_gt.xyz[:2])
        target_xy = np.array(scene.target_pose_gt.xyz[:2])
        _, g0, g1 = next(s for s in self.phase_spans if s[0] is Phase.GRASP)
        _, _, ins_end = next(s for s in self.phase_spans if s[0] is Phase.INSERT)
        grasp_err = float(np.linalg.norm(cmd[g1 - 1, :2] - part_xy))
        align_err = float(np.linalg.norm(actual[ins_end - 1, :2] - target_xy))
        if grasp_err >= GRASP_TOL_M:
            success, phase_reached, failure_phase, failure_label = \
                False, Phase.APPROACH, Phase.GRASP, "grasp_miss"
        elif align_err >= gap:
            success, phase_reached, failure_phase, failure_label = \
                False, Phase.ALIGN, Phase.INSERT, "insertion_misalign"
        else:
            success, phase_reached, failure_phase, failure_label = True, Phase.DONE, None, None

        wrench = np.zeros((n, 6))
        for ph in (Phase.INSERT, Phase.FASTEN):
            _, i0, i1 = next(s for s in self.phase_spans if s[0] is ph)
            lat = actual[i0:i1, :2] - target_xy[None, :]
            lat_n = np.linalg.norm(lat, axis=1)
            over = np.clip(lat_n - gap, 0.0, None)
            unit = np.where(lat_n[:, None] > 0, lat / np.maximum(lat_n[:, None], 1e-12), 0.0)
            wrench[i0:i1, 0:2] = FORCE_K * over[:, None] * unit
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
            episode_id=f"{scene.scene_id}__{model_id}__{hw.hw_config_id}__agx__s{seed}",
            benchmark_version=BENCHMARK_VERSION + "+agx", source=Source.SIM,
            scene=scene,
            model=ModelTrace(model_id=model_id,
                             language_instruction="pick the cap from the bin and screw it onto the bottle",
                             chunk=chunk),
            robot=robot, outcome=outcome,
            media={"video_uris": [], "sim_state_log_uri": None},
        )