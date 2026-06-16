"""FR-2.4 standard model interface — real implementation (the interface itself).
To wire up a real VLA: implement a VLAModel subclass and call a remote gRPC/HTTP endpoint inside infer();
the rest of the evaluation side needs zero changes. uncertainty is optional; when absent the related metrics record N/A (FR-2.4 / D5).

Vertical-slice simplification: infer returns the whole command trajectory at once (open-loop action chunk);
closed-loop chunked inference extends the same interface (chunk concatenation), listed as a known limitation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..schema import ActionChunk, Phase, Scene


@dataclass
class Observation:
    """The observation fed to the model. In a real system this would be images + proprioception; in the fake
    implementation the model simulates its own perception error, so we pass scene ground truth and lighting through to
    the (fake) model — replace with rendered images when wiring up a real model."""
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