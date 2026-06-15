"""MuJoCo 物理后端 —— Backend 接口的真实现 (rq.md O-1, M2+)。

物理建模 (本版边界):
- 末端执行器 = 4-DOF 工具体 (slide x/y/z + hinge yaw), 位置伺服驱动;
  跟踪迟滞/超调由真实二阶动力学产生, 不再是解析滤波。
- 硬件档案 → 伺服刚度/阻尼 + 注入扰动力 (恒力=漂移, 正弦力=抖动, 随机力=噪声)。
- 抓取/拧紧的接触动力学不在本版范围 (成功判据与 FakeSim 同口径, 便于跨后端对比);
  接触级保真度列入 future work。

import 失败时优雅降级: MUJOCO_AVAILABLE=False, 实例化给出明确指引。
"""
from __future__ import annotations

import numpy as np

from ..schema import (BENCHMARK_VERSION, Episode, ModelTrace, Outcome, Phase,
                      RobotTrace, Scene, Source)
from ..models.base import Observation, VLAModel
from .base import Backend, HardwareProfile
from .fake_sim import DT, FORCE_K, GRASP_TOL_M, _phase_spans

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

_HOME = np.array([0.30, 0.0, 0.30])

# 硬件标定档案 → MuJoCo 伺服/扰动参数 (droop/jitter/noise 单位: N)
MJ_PROFILES = {
    "arm-calibrated-2026Q2": dict(kp=2000.0, kv=80.0, droop=(0.10, -0.06),
                                  jitter_f=0.30, noise_f=0.15),
    "arm-worn-2023Q1": dict(kp=300.0, kv=60.0, droop=(0.25, -0.14),
                            jitter_f=2.80, noise_f=1.00),
    "arm-pristine": dict(kp=3000.0, kv=100.0, droop=(0.04, -0.02),
                         jitter_f=0.12, noise_f=0.08),
}
JITTER_FREQS = (12.0, 27.0)
PHYS_DT = 0.002  # 物理步长; 100 Hz 控制 → 每控制步 5 个物理子步


def _build_xml(kp: float, kv: float) -> str:
    return f"""
<mujoco>
  <option timestep="{PHYS_DT}" gravity="0 0 0"/>
  <worldbody>
    <body name="tool" pos="{_HOME[0]} {_HOME[1]} {_HOME[2]}">
      <joint name="jx" type="slide" axis="1 0 0" damping="{kv}"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="{kv}"/>
      <joint name="jz" type="slide" axis="0 0 1" damping="{kv}"/>
      <joint name="jyaw" type="hinge" axis="0 0 1" damping="2"/>
      <geom type="sphere" size="0.01" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="jx" kp="{kp}"/>
    <position joint="jy" kp="{kp}"/>
    <position joint="jz" kp="{kp}"/>
    <position joint="jyaw" kp="50"/>
  </actuator>
</mujoco>"""


class MujocoBackend(Backend):
    def __init__(self, dt: float = DT):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError(
                "mujoco 未安装。pip install mujoco 后使用; "
                "当前环境请用 FakeSimBackend (接口相同)。")
        self.dt = dt
        self.substeps = int(round(dt / PHYS_DT))
        self.phase_spans, self.n_steps = _phase_spans(dt)
        self.t_grid = np.arange(self.n_steps) * dt
        self._models = {}  # hw_config_id -> (MjModel, tool_body_id)

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

    def _mj(self, hw: HardwareProfile):
        if hw.hw_config_id not in self._models:
            prof = MJ_PROFILES[hw.hw_config_id]
            m = mujoco.MjModel.from_xml_string(_build_xml(prof["kp"], prof["kv"]))
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")
            self._models[hw.hw_config_id] = (m, bid, prof)
        return self._models[hw.hw_config_id]

    def _execute(self, scene: Scene, model_id: str, chunk, hw, seed: int) -> Episode:
        m, bid, prof = self._mj(hw)
        d = mujoco.MjData(m)
        rng_hw = np.random.default_rng(seed + 10_000_019)
        phases = rng_hw.uniform(0, 2 * np.pi, (len(JITTER_FREQS), 2))
        n = self.n_steps
        cmd = chunk.cmd_xyz
        actual = np.zeros((n, 3))
        yaw_actual = np.zeros(n)

        droop = np.array(prof["droop"])
        for i in range(n):
            d.ctrl[0:3] = cmd[i] - _HOME          # 位置伺服目标 (关节系)
            d.ctrl[3] = chunk.cmd_yaw[i]
            # 扰动力: 漂移 + 抖动 + 噪声 (作用于工具体 xy)
            f = droop.copy()
            for k, fr in enumerate(JITTER_FREQS):
                f += prof["jitter_f"] * np.sin(2 * np.pi * fr * self.t_grid[i] + phases[k])
            f += rng_hw.normal(0.0, prof["noise_f"], 2)
            d.xfrc_applied[bid, 0:2] = f
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
            actual[i] = _HOME + d.qpos[0:3]
            yaw_actual[i] = d.qpos[3]

        # 成功判据与 wrench/夹爪 —— 与 FakeSim 同口径 (跨后端可比)
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
            episode_id=f"{scene.scene_id}__{model_id}__{hw.hw_config_id}__mj__s{seed}",
            benchmark_version=BENCHMARK_VERSION + "+mujoco", source=Source.SIM,
            scene=scene,
            model=ModelTrace(model_id=model_id,
                             language_instruction="从料箱拿起瓶盖, 拧紧到瓶子上",
                             chunk=chunk),
            robot=robot, outcome=outcome,
            media={"video_uris": [], "sim_state_log_uri": None},
        )
