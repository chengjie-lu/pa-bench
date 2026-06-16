"""Execution-backend abstraction (rq.md §9 runners layer) — real implementation (interface).
MuJoCo / real-robot backends implement the same Backend interface for plug-and-play (NFR-5)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..schema import Episode, Scene
from ..models.base import VLAModel


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware calibration profile (the object referenced by rq.md §5 hw_config_id)."""
    hw_config_id: str
    tracking_alpha: float      # first-order tracking-filter coefficient (smaller = more sluggish)
    droop_xy: tuple            # systematic tracking offset [m] (wear / calibration drift)
    track_noise_std: float     # tracking white noise [m]
    jitter_amp: float          # end-effector jitter amplitude [m] (5–50 Hz band)
    jitter_freqs: tuple = (12.0, 27.0)


class Backend(ABC):
    @abstractmethod
    def run_episode(self, scene: Scene, model: VLAModel,
                    hw: HardwareProfile, seed: int) -> Episode: ...

    @abstractmethod
    def run_oracle(self, scene: Scene, hw: HardwareProfile, seed: int) -> Episode:
        """FR-2.5 Oracle replay: a perfect-perception expert trajectory executed on the
        same hardware/scene, used as the attribution control."""