"""AGX Dynamics 物理后端 —— Backend 接口的真实现 (rq.md O-1, NFR-5 即插即用)。

为什么再加一个物理后端:
- MuJoCo 偏研究/快速原型; AGX Dynamics (Algoryx) 是工业级、商用许可的物理引擎,
  在约束求解 (直接求解器, 非弹簧近似)、接触保真、与数字孪生工具链对接上更贴近
  真实产线 —— 精密装配评测的高保真档位。
- 三后端 (Fake / MuJoCo / AGX) 实现同一 `Backend` 接口 → 上层 pipeline / API / 报告
  零改动即可切换 (见 RunConfig.backend, BACKEND_REGISTRY)。

物理建模 (与 MuJoCo 后端同口径, 保证跨后端排序可比 —— crossbackend_eval 思路):
- 末端执行器 = 笛卡尔龙门 (gantry): world →[prismatic X]→ bx →[prismatic Y]→ by
  →[prismatic Z]→ bz →[hinge yaw]→ tool, 共 4-DOF, 与 MuJoCo 的 slide×3+hinge 对应。
- 位置伺服 = 每个约束的 LockController, 柔度 compliance=1/kp, 阻尼=kv;
  AGX 直接求解器产生的跟踪迟滞/超调是真实约束动力学, 不是解析滤波。
- 硬件档案 (HardwareProfile.hw_config_id) → 伺服刚度/阻尼 + 注入扰动力
  (恒力=漂移 droop, 正弦力=抖动 jitter, 高斯力=噪声), 力施加在 tool 体的 xy。
- 抓取/拧紧的接触动力学不在本版范围 (成功判据、wrench、夹爪与 FakeSim/MuJoCo 同口径),
  接触级保真 (AGX 的强项) 列入 future work。

import / 许可不可用时优雅降级: AGX_AVAILABLE=False, 实例化给出明确指引。
AGX 是商用许可产品 (需 license server / 离线许可文件), CI 与本机通常缺失 →
对应测试 pytest.mark.skipif 跳过, 与 MuJoCo 后端处理一致。
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

# 硬件标定档案 → AGX 伺服/扰动参数。
# kp [N/m] 越大越刚 (compliance=1/kp); kv [N·s/m] 关节阻尼; droop/jitter/noise 单位 N。
# 数值与 runners/mujoco_sim.MJ_PROFILES 对齐, 使两物理后端的 SR 排序可直接比对。
AGX_PROFILES = {
    "arm-calibrated-2026Q2": dict(kp=2000.0, kv=80.0, droop=(0.10, -0.06),
                                  jitter_f=0.30, noise_f=0.15),
    "arm-worn-2023Q1": dict(kp=300.0, kv=60.0, droop=(0.25, -0.14),
                            jitter_f=2.80, noise_f=1.00),
    "arm-pristine": dict(kp=3000.0, kv=100.0, droop=(0.04, -0.02),
                         jitter_f=0.12, noise_f=0.08),
}
JITTER_FREQS = (12.0, 27.0)
PHYS_DT = 0.002  # 物理步长; 100 Hz 控制 → 每控制步 5 个物理子步 (与 MuJoCo 同)


class AgxBackend(Backend):
    def __init__(self, dt: float = DT):
        if not AGX_AVAILABLE:
            raise RuntimeError(
                "agx (AGX Dynamics) 未安装或许可不可用。\n"
                "  安装: 见 Algoryx AGX Dynamics 发行版的 Python 绑定 (agxpy);\n"
                "  许可: 设置 AGX_LICENSE / 许可文件后导入 `agx` 应成功;\n"
                "  当前环境请用 FakeSimBackend 或 MujocoBackend (接口相同)。")
        # AGX 全局运行时只需初始化一次 (idempotent)。
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
                          instruction="从料箱拿起瓶盖, 拧紧到瓶子上",
                          t_grid=self.t_grid, phase_spans=self.phase_spans)
        chunk = model.infer(obs, rng_model)
        return self._execute(scene, model.model_id, chunk, hw, seed)

    def run_oracle(self, scene: Scene, hw: HardwareProfile, seed: int) -> Episode:
        from .fake_sim import FakeSimBackend
        chunk = FakeSimBackend(self.dt)._plan_oracle(scene)  # 同一专家轨迹规划器
        return self._execute(scene, "oracle-replay", chunk, hw, seed)

    # ------------------------------------------------------------ internals

    def _build_world(self, prof: dict):
        """构建 4-DOF 龙门 + 伺服, 返回 (sim, tool_body, [lock_x, lock_y, lock_z, lock_yaw])。

        每次回合新建独立 Simulation, 保证确定性 (无跨回合状态泄漏)。
        """
        sim = agxSDK.Simulation()
        sim.setTimeStep(PHYS_DT)
        sim.setUniformGravity(agx.Vec3(0.0, 0.0, 0.0))  # 与 MuJoCo gravity=0 同, 隔离重力影响

        def _slider(mass: float, pos: np.ndarray) -> "agx.RigidBody":
            rb = agx.RigidBody()
            rb.add(agxCollide.Geometry(agxCollide.Sphere(0.01)))
            rb.getMassProperties().setMass(mass)
            rb.setPosition(agx.Vec3(float(pos[0]), float(pos[1]), float(pos[2])))
            sim.add(rb)
            return rb

        # 龙门链: 各级承载下游, 末级 tool 质量 1 kg (与 MuJoCo 工具体一致)。
        bx = _slider(1.0, _HOME)
        by = _slider(1.0, _HOME)
        bz = _slider(1.0, _HOME)
        tool = _slider(1.0, _HOME)

        kp, kv = prof["kp"], prof["kv"]
        compliance = 1.0 / kp

        def _prismatic(axis, rb1, rb2):
            # rb1 沿世界 axis 相对 rb2 平移; rb2=None → 相对世界。
            pris = agx.Prismatic(agx.Vec3(*axis), rb1, rb2) if rb2 is not None \
                else agx.Prismatic(agx.Vec3(*axis), rb1)
            pris.getMotor1D().setEnable(False)
            lock = pris.getLock1D()
            lock.setEnable(True)
            lock.setCompliance(compliance)   # 柔度=1/kp → 刚度 kp
            lock.setDamping(kv * PHYS_DT)     # AGX 阻尼以 (阻尼系数·dt) 计
            sim.add(pris)
            return lock

        lock_x = _prismatic((1, 0, 0), bx, None)
        lock_y = _prismatic((0, 1, 0), by, bx)
        lock_z = _prismatic((0, 0, 1), bz, by)

        # yaw: tool 绕世界 Z 相对 bz 旋转
        hinge = agx.Hinge(agx.Vec3(0, 0, 1), tool, bz)
        hinge.getMotor1D().setEnable(False)
        lock_yaw = hinge.getLock1D()
        lock_yaw.setEnable(True)
        lock_yaw.setCompliance(1.0 / 50.0)   # yaw 伺服刚度 (与 MuJoCo kp=50 对齐)
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
            # 位置伺服目标 (关节系 = 相对 HOME 的位移)
            tgt = cmd[i] - _HOME
            lx.setPosition(float(tgt[0]))
            ly.setPosition(float(tgt[1]))
            lz.setPosition(float(tgt[2]))
            lyaw.setPosition(float(chunk.cmd_yaw[i]))
            # 扰动力: 漂移 + 抖动 + 噪声 (作用于 tool 体 xy)
            f = droop.copy()
            for k, fr in enumerate(JITTER_FREQS):
                f += prof["jitter_f"] * np.sin(2 * np.pi * fr * self.t_grid[i] + phases[k])
            f += rng_hw.normal(0.0, prof["noise_f"], 2)
            for _ in range(self.substeps):
                tool.setForce(agx.Vec3(float(f[0]), float(f[1]), 0.0))
                sim.stepForward()
            p = tool.getPosition()
            actual[i] = (p.x(), p.y(), p.z())
            # tool 绕 Z 的旋转角 (四元数 → yaw)
            rot = tool.getRotation()
            yaw_actual[i] = float(np.arctan2(
                2.0 * (rot.w() * rot.z() + rot.x() * rot.y()),
                1.0 - 2.0 * (rot.y() ** 2 + rot.z() ** 2)))

        # 成功判据与 wrench/夹爪 —— 与 FakeSim / MuJoCo 同口径 (跨后端可比)
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
                             language_instruction="从料箱拿起瓶盖, 拧紧到瓶子上",
                             chunk=chunk),
            robot=robot, outcome=outcome,
            media={"video_uris": [], "sim_state_log_uri": None},
        )
