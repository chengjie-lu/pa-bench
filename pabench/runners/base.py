"""执行后端抽象 (rq.md §9 runners 层) —— 真实现 (接口)。
MuJoCo / 真机后端实现同一 Backend 接口即插即用 (NFR-5)。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schema import Episode, Scene
from ..models.base import VLAModel


@dataclass(frozen=True)
class HardwareProfile:
    """硬件标定档案 (rq.md §5 hw_config_id 指向的对象)。"""
    hw_config_id: str
    tracking_alpha: float      # 一阶跟踪滤波系数 (越小越迟钝)
    droop_xy: tuple            # 系统性跟踪偏移 [m] (磨损/标定漂移)
    track_noise_std: float     # 跟踪白噪声 [m]
    jitter_amp: float          # 末端抖动幅值 [m] (5–50 Hz 频段)
    jitter_freqs: tuple = (12.0, 27.0)


class Backend(ABC):
    @abstractmethod
    def run_episode(self, scene: Scene, model: VLAModel,
                    hw: HardwareProfile, seed: int) -> Episode: ...

    @abstractmethod
    def run_oracle(self, scene: Scene, hw: HardwareProfile, seed: int) -> Episode:
        """FR-2.5 Oracle 回放: 完美感知的专家轨迹在同硬件/同场景执行, 归因对照用。"""