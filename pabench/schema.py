"""Core data contract (rq.md §5) — all real implementation, the single contract for every module.

Vertical-slice simplifications (forward-compatible, no impact on field layout):
- SE3Pose rotation keeps only the yaw about the world z axis; full SO(3) is left for extension.
- The robot-telemetry schema uses fixed-length array columns instead of per-step ActionStep objects, with semantics matching rq.md.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

BENCHMARK_VERSION = "pa-bench-0.1.0+slice-m1"


# ---------------------------------------------------------------- enums


class TaskType(str, Enum):
    PICK = "pick"
    PLACE_IN_FIXTURE = "place_in_fixture"
    PEG_INSERT = "peg_insert"
    SCREW_CAP = "screw_cap"
    SNAP_FIT = "snap_fit"


class ToleranceClass(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"

    @property
    def gap_m(self) -> float:
        """Assembly fit clearance (rq.md §2.4)."""
        return {"T1": 1.0e-3, "T2": 0.5e-3, "T3": 0.2e-3}[self.value]


class Source(str, Enum):
    SIM = "sim"
    REAL = "real"


class GenerationMethod(str, Enum):
    NOMINAL = "nominal"
    MUTATION = "mutation"
    METAMORPHIC = "metamorphic"
    ADVERSARIAL_OPT = "adversarial_opt"
    REAL_SAMPLE = "real_sample"


class Phase(str, Enum):
    APPROACH = "approach"
    GRASP = "grasp"
    TRANSFER = "transfer"
    ALIGN = "align"
    INSERT = "insert"
    FASTEN = "fasten"
    DONE = "done"


class Attribution(str, Enum):
    MODEL = "model"
    HARDWARE = "hardware"
    ENVIRONMENT = "environment"
    AMBIGUOUS = "ambiguous"


# ---------------------------------------------------------------- pose


@dataclass(frozen=True)
class SE3Pose:
    xyz: tuple  # (x, y, z) [m]
    yaw: float  # [rad], about the world z axis

    def rotated_z(self, theta: float, about=(0.0, 0.0)) -> "SE3Pose":
        c, s = math.cos(theta), math.sin(theta)
        x = self.xyz[0] - about[0]
        y = self.xyz[1] - about[1]
        return SE3Pose(
            (about[0] + c * x - s * y, about[1] + s * x + c * y, self.xyz[2]),
            self.yaw + theta,
        )

    def to_dict(self):
        return {"xyz": [round(float(v), 9) for v in self.xyz], "yaw": round(float(self.yaw), 9)}

    @staticmethod
    def from_dict(d):
        return SE3Pose(tuple(d["xyz"]), d["yaw"])


# ---------------------------------------------------------------- scene


@dataclass
class Scene:
    scene_id: str
    task_type: TaskType
    tolerance_class: ToleranceClass
    part_pose_gt: SE3Pose
    target_pose_gt: SE3Pose
    perturbation: dict = field(default_factory=dict)
    generation_method: GenerationMethod = GenerationMethod.NOMINAL
    parent_episode_id: Optional[str] = None
    mr_id: Optional[str] = None

    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "task_type": self.task_type.value,
            "tolerance_class": self.tolerance_class.value,
            "part_pose_gt": self.part_pose_gt.to_dict(),
            "target_pose_gt": self.target_pose_gt.to_dict(),
            "perturbation": {k: round(float(v), 9) if isinstance(v, float) else v
                             for k, v in sorted(self.perturbation.items())},
            "generation_method": self.generation_method.value,
            "parent_episode_id": self.parent_episode_id,
            "mr_id": self.mr_id,
        }

    @staticmethod
    def from_dict(d):
        return Scene(
            scene_id=d["scene_id"],
            task_type=TaskType(d["task_type"]),
            tolerance_class=ToleranceClass(d["tolerance_class"]),
            part_pose_gt=SE3Pose.from_dict(d["part_pose_gt"]),
            target_pose_gt=SE3Pose.from_dict(d["target_pose_gt"]),
            perturbation=dict(d["perturbation"]),
            generation_method=GenerationMethod(d["generation_method"]),
            parent_episode_id=d["parent_episode_id"],
            mr_id=d["mr_id"],
        )


# ---------------------------------------------------------------- model trace


def _arr(a, nd):
    return None if a is None else np.asarray(a, dtype=float).reshape((-1,) if nd == 1 else (-1, nd))


def _arr_dict(a):
    return None if a is None else np.round(np.asarray(a, dtype=float), 9).tolist()


@dataclass
class ActionChunk:
    """Action chunk emitted by the model (rq.md FR-2.4): command trajectory + optional uncertainty + inference latency."""
    t: np.ndarray            # (N,) [s]
    cmd_xyz: np.ndarray      # (N,3) [m]
    cmd_yaw: np.ndarray      # (N,) [rad]
    entropy: Optional[np.ndarray]  # (N,) optional — when absent, related metrics record N/A (FR-2.4)
    latency_ms: np.ndarray   # (N,)

    def to_dict(self):
        return {
            "t": _arr_dict(self.t),
            "cmd_xyz": _arr_dict(self.cmd_xyz),
            "cmd_yaw": _arr_dict(self.cmd_yaw),
            "entropy": _arr_dict(self.entropy),
            "latency_ms": _arr_dict(self.latency_ms),
        }

    @staticmethod
    def from_dict(d):
        return ActionChunk(
            t=_arr(d["t"], 1), cmd_xyz=_arr(d["cmd_xyz"], 3), cmd_yaw=_arr(d["cmd_yaw"], 1),
            entropy=_arr(d["entropy"], 1), latency_ms=_arr(d["latency_ms"], 1),
        )


@dataclass
class ModelTrace:
    model_id: str
    language_instruction: str
    chunk: ActionChunk

    def to_dict(self):
        return {"model_id": self.model_id,
                "language_instruction": self.language_instruction,
                "actions": self.chunk.to_dict()}

    @staticmethod
    def from_dict(d):
        return ModelTrace(d["model_id"], d["language_instruction"], ActionChunk.from_dict(d["actions"]))


# ---------------------------------------------------------------- robot trace


@dataclass
class RobotTrace:
    """Hardware telemetry (rq.md §5 robot), 100 Hz uniform time grid (FR-2.3: command and telemetry share the grid → zero alignment error)."""
    t: np.ndarray              # (N,)
    ee_xyz_actual: np.ndarray  # (N,3) measured end-effector position
    ee_yaw_actual: np.ndarray  # (N,)
    ft_wrench: np.ndarray      # (N,6) [Fx,Fy,Fz,Tx,Ty,Tz] — this slice fills only Fx,Fy
    gripper_width: np.ndarray  # (N,)
    hw_config_id: str
    phase_spans: list = field(default_factory=list)  # [(Phase, i0, i1)] rule-based segmentation result (FR-3.4)

    def to_dict(self):
        return {
            "t": _arr_dict(self.t),
            "ee_xyz_actual": _arr_dict(self.ee_xyz_actual),
            "ee_yaw_actual": _arr_dict(self.ee_yaw_actual),
            "ft_wrench": _arr_dict(self.ft_wrench),
            "gripper_width": _arr_dict(self.gripper_width),
            "hw_config_id": self.hw_config_id,
            "phase_spans": [[p.value, int(i0), int(i1)] for p, i0, i1 in self.phase_spans],
        }

    @staticmethod
    def from_dict(d):
        return RobotTrace(
            t=_arr(d["t"], 1), ee_xyz_actual=_arr(d["ee_xyz_actual"], 3),
            ee_yaw_actual=_arr(d["ee_yaw_actual"], 1), ft_wrench=_arr(d["ft_wrench"], 6),
            gripper_width=_arr(d["gripper_width"], 1), hw_config_id=d["hw_config_id"],
            phase_spans=[(Phase(p), i0, i1) for p, i0, i1 in d["phase_spans"]],
        )

    def span(self, phase: Phase):
        for p, i0, i1 in self.phase_spans:
            if p is phase:
                return i0, i1
        raise KeyError(phase)


# ---------------------------------------------------------------- outcome / episode


@dataclass
class Outcome:
    success: bool
    phase_reached: Phase
    failure_phase: Optional[Phase]
    failure_label: Optional[str]
    attribution: Optional[Attribution]
    duration_s: float
    attribution_reason: Optional[str] = None  # slice addition: the attribution decision-tree rule that fired

    def to_dict(self):
        return {
            "success": self.success,
            "phase_reached": self.phase_reached.value,
            "failure_phase": self.failure_phase.value if self.failure_phase else None,
            "failure_label": self.failure_label,
            "attribution": self.attribution.value if self.attribution else None,
            "attribution_reason": self.attribution_reason,
            "duration_s": round(float(self.duration_s), 9),
        }

    @staticmethod
    def from_dict(d):
        return Outcome(
            success=d["success"], phase_reached=Phase(d["phase_reached"]),
            failure_phase=Phase(d["failure_phase"]) if d["failure_phase"] else None,
            failure_label=d["failure_label"],
            attribution=Attribution(d["attribution"]) if d["attribution"] else None,
            duration_s=d["duration_s"], attribution_reason=d["attribution_reason"],
        )


@dataclass
class Episode:
    episode_id: str
    benchmark_version: str
    source: Source
    scene: Scene
    model: ModelTrace
    robot: RobotTrace
    outcome: Outcome
    media: dict = field(default_factory=dict)  # video_uris / sim_state_log_uri — real-robot/render hookup point

    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "benchmark_version": self.benchmark_version,
            "source": self.source.value,
            "scene": self.scene.to_dict(),
            "model": self.model.to_dict(),
            "robot": self.robot.to_dict(),
            "outcome": self.outcome.to_dict(),
            "media": self.media,
        }

    @staticmethod
    def from_dict(d):
        return Episode(
            episode_id=d["episode_id"], benchmark_version=d["benchmark_version"],
            source=Source(d["source"]), scene=Scene.from_dict(d["scene"]),
            model=ModelTrace.from_dict(d["model"]), robot=RobotTrace.from_dict(d["robot"]),
            outcome=Outcome.from_dict(d["outcome"]), media=dict(d["media"]),
        )

    def content_hash(self) -> str:
        """NFR-1: replaying with the same benchmark_version+seed → identical hash."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------- store (FR-1.6 lineage machine-check)


class LineageError(ValueError):
    pass


class EpisodeStore:
    """Validate lineage on write (FR-1.6): non-nominal episodes must carry parent_episode_id;
    metamorphic episodes must also carry mr_id."""

    def __init__(self):
        self.episodes: list[Episode] = []

    def add(self, ep: Episode):
        gm = ep.scene.generation_method
        if gm is not GenerationMethod.NOMINAL and not ep.scene.parent_episode_id:
            raise LineageError(f"{ep.episode_id}: generation_method={gm.value} but missing parent_episode_id")
        if gm is GenerationMethod.METAMORPHIC and not ep.scene.mr_id:
            raise LineageError(f"{ep.episode_id}: metamorphic episode missing mr_id")
        self.episodes.append(ep)

    def save_jsonl(self, path):
        with open(path, "w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep.to_dict(), sort_keys=True) + "\n")

    def __len__(self):
        return len(self.episodes)