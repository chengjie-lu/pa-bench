import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pabench.models import PreciseVLA, SloppyVLA  # noqa: E402
from pabench.runners import CALIBRATED_ARM, WORN_ARM, FakeSimBackend  # noqa: E402
from pabench.scenegen import MutationGenerator, nominal_screw_cap  # noqa: E402


@pytest.fixture(scope="session")
def backend():
    return FakeSimBackend()


@pytest.fixture(scope="session")
def base_scene():
    return nominal_screw_cap()


@pytest.fixture(scope="session")
def mutated_scenes(base_scene):
    return [base_scene] + MutationGenerator(seed=11).generate(
        base_scene, f"{base_scene.scene_id}__anchor", 29)


def run_batch(backend, model, hw, scenes, seed_base):
    return [backend.run_episode(s, model, hw, seed=seed_base + i)
            for i, s in enumerate(scenes)]


@pytest.fixture(scope="session")
def precise_calibrated(backend, mutated_scenes):
    return run_batch(backend, PreciseVLA(), CALIBRATED_ARM, mutated_scenes, 5000)


@pytest.fixture(scope="session")
def precise_worn(backend, mutated_scenes):
    return run_batch(backend, PreciseVLA(), WORN_ARM, mutated_scenes, 5000)


@pytest.fixture(scope="session")
def sloppy_calibrated(backend, mutated_scenes):
    return run_batch(backend, SloppyVLA(), CALIBRATED_ARM, mutated_scenes, 5000)