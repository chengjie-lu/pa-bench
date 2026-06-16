"""Episode contract: serialization round-trip, hashing, lineage machine-check (FR-1.6 / NFR-1)."""
import pytest

from pabench.schema import Episode, EpisodeStore, GenerationMethod, LineageError, Scene, SE3Pose, TaskType, ToleranceClass


def test_episode_roundtrip_preserves_content(precise_calibrated):
    ep = precise_calibrated[0]
    d = ep.to_dict()
    ep2 = Episode.from_dict(d)
    assert ep2.to_dict() == d
    assert ep2.content_hash() == ep.content_hash()


def test_tolerance_gap_values():
    assert ToleranceClass.T1.gap_m == pytest.approx(1.0e-3)
    assert ToleranceClass.T3.gap_m < ToleranceClass.T2.gap_m < ToleranceClass.T1.gap_m


def test_store_rejects_orphan_mutation(precise_calibrated):
    """FR-1.6: a non-nominal episode missing parent_episode_id must be rejected on write."""
    ep = Episode.from_dict(precise_calibrated[0].to_dict())
    ep.scene.generation_method = GenerationMethod.MUTATION
    ep.scene.parent_episode_id = None
    store = EpisodeStore()
    with pytest.raises(LineageError):
        store.add(ep)


def test_store_rejects_metamorphic_without_mr_id(precise_calibrated):
    ep = Episode.from_dict(precise_calibrated[0].to_dict())
    ep.scene.generation_method = GenerationMethod.METAMORPHIC
    ep.scene.parent_episode_id = "some-parent"
    ep.scene.mr_id = None
    with pytest.raises(LineageError):
        EpisodeStore().add(ep)


def test_se3_rotation_z():
    p = SE3Pose((1.0, 0.0, 0.5), 0.0)
    import math
    q = p.rotated_z(math.pi / 2)
    assert q.xyz[0] == pytest.approx(0.0, abs=1e-12)
    assert q.xyz[1] == pytest.approx(1.0)
    assert q.xyz[2] == pytest.approx(0.5)
    assert q.yaw == pytest.approx(math.pi / 2)