"""FR-2.4 标准模型接口 —— 真实现 (接口本身)。
接真实 VLA 时: 实现一个 VLAModel 子类, 在 infer() 里调远端 gRPC/HTTP 端点即可,
评测侧其余代码零改动。uncertainty 可选, 缺失时相关指标记 N/A (FR-2.4 / D5)。

纵切简化: infer 一次返回整条指令轨迹 (open-loop action chunk);
闭环分块推理按同一接口扩展 (chunk 拼接), 列入已知限制。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..schema import ActionChunk, Phase, Scene


@dataclass
class Observation:
    """喂给模型的观测。真实系统中这里是图像+本体; 假实现中模型自行模拟感知误差,
    因此把场景真值与光照透传给(假)模型 —— 真模型接入时换成渲染图像。"""
    scene: Scene
    lux_factor: float
    instruction: str
    t_grid: np.ndarray                # (N,) 100 Hz
    phase_spans: list                 # [(Phase, i0, i1)]


class VLAModel(ABC):
    model_id: str = "abstract"

    @abstractmethod
    def infer(self, obs: Observation, rng: np.random.Generator) -> ActionChunk:
        """infer(images, instruction, proprio) -> {action_chunk, uncertainty?}"""