"""【桩 / FAKE】两个脚本化假 VLA 模型 —— 替代真实模型权重 (本环境装不下/跑不动)。

它们实现 VLAModel 接口, 行为可控且互相区分, 用于驱动整条评测链路:
- PreciseVLA: 小感知噪声、min-jerk 平滑轨迹、坐标系等变 (MR-1 应通过)、
  不确定性输出与其真实感知误差强相关 (AUROC 应高)。
- SloppyVLA : 世界系固定感知偏置 (→ 不等变, MR-1 应违反) + 大噪声 + 逐步抖动
  (jerk 应高), 不确定性输出无信息量。

接真实模型时: 新建 VLAModel 子类调用远端推理服务, 删除/绕过本文件即可。
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
    """共享的轨迹脚本: 感知(带各自误差模型) → 按阶段生成 waypoint → 插值成指令轨迹。"""

    sigma_base_m: float = 0.0          # 感知噪声 (1/lux 放大, 模拟暗光退化)
    bias_world_m = np.zeros(2)         # 世界系固定偏置 (非等变来源)
    wobble_m: float = 0.0              # 指令逐步抖动 (jerk 来源)
    smooth = True

    def _perceive(self, pose: SE3Pose, lux: float, rng) -> np.ndarray:
        sigma = self.sigma_base_m / max(lux, 1e-6)
        err = self.bias_world_m + rng.normal(0.0, sigma, 2)
        return np.array([pose.xyz[0] + err[0], pose.xyz[1] + err[1], pose.xyz[2]]), err

    def _entropy(self, n: int, target_err: np.ndarray, rng) -> np.ndarray:
        raise NotImplementedError  # 子类各自的不确定性头 (桩内允许)

    def _latency(self, n: int, rng) -> np.ndarray:
        raise NotImplementedError

    def infer(self, obs: Observation, rng: np.random.Generator) -> ActionChunk:
        part_hat, _part_err = self._perceive(obs.scene.part_pose_gt, obs.lux_factor, rng)
        target_hat, target_err = self._perceive(obs.scene.target_pose_gt, obs.lux_factor, rng)

        # 阶段终点 waypoint (全部基于"感知到"的位姿 → 场景旋转时自然等变, 除非有世界系偏置)
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
            if phase is Phase.FASTEN:  # 拧紧: yaw 旋转 2π
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
        # 校准良好: 熵 ∝ 本回合真实感知误差 (模拟 ensemble 方差对错误的敏感)
        return 0.3 + 3000.0 * float(np.linalg.norm(target_err)) + 0.05 * rng.standard_normal(n)

    def _latency(self, n, rng):
        return np.clip(rng.normal(12.0, 2.0, n), 1.0, None)


class SloppyVLA(_ScriptedVLA):
    model_id = "sloppy-vla-0.1"
    sigma_base_m = 1.0e-3
    bias_world_m = 1.2e-3 * np.array([math.cos(0.7), math.sin(0.7)])  # 世界系偏置 → MR-1 违反
    wobble_m = 0.3e-3
    smooth = False

    def _entropy(self, n, target_err, rng):
        # 无信息量: 与真实误差无关
        return 1.0 + 0.1 * rng.standard_normal(n)

    def _latency(self, n, rng):
        return np.clip(rng.normal(35.0, 8.0, n), 1.0, None)