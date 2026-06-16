"""Attribution engine: every decision-tree branch (FR-4.1) + end-to-end attribution behavior (FR-4.2/4.3)."""
import pytest

from pabench.attribution import AttributionThresholds, attribute_episode, decide
from pabench.schema import Attribution

TH = AttributionThresholds()


# ---------------- pure-function decision-tree branch coverage


def test_rule1_mr_violation_is_model():
    a, r = decide(True, 0.5, 0.0001, 1.0, TH)
    assert a is Attribution.MODEL and "rule1" in r


def test_rule2_bad_plan_good_track_is_model():
    a, r = decide(False, 1.8, 0.0001, 1.0, TH)
    assert a is Attribution.MODEL and "rule2" in r


def test_rule3_good_plan_bad_track_is_hardware():
    a, r = decide(False, 0.4, 0.0009, 1.0, TH)
    assert a is Attribution.HARDWARE and "rule3" in r


def test_rule4_oracle_succeeds_is_model():
    a, r = decide(False, 1.8, 0.0009, 1.0, TH, oracle_success=True)
    assert a is Attribution.MODEL and "rule4a" in r


def test_rule4_oracle_fails_ood_is_environment():
    a, r = decide(False, 1.8, 0.0009, 0.32, TH, oracle_success=False)
    assert a is Attribution.ENVIRONMENT and "rule4b" in r


def test_rule4_oracle_fails_in_dist_is_hardware():
    a, r = decide(False, 1.8, 0.0009, 0.9, TH, oracle_success=False)
    assert a is Attribution.HARDWARE and "rule4c" in r


def test_rule4_no_oracle_is_ambiguous():
    a, r = decide(False, 1.8, 0.0009, 0.9, TH, oracle_success=None)
    assert a is Attribution.AMBIGUOUS and "rule4" in r


# ---------------- end-to-end: attribution results should match how the failure was injected


def _attribute_all(backend, episodes, hw):
    counts = {}
    for ep in episodes:
        def oracle_fn(e):
            seed = int(e.episode_id.rsplit("__s", 1)[1])
            return backend.run_oracle(e.scene, hw, seed).outcome.success
        a = attribute_episode(ep, TH, oracle_fn=oracle_fn)
        if a:
            counts[a] = counts.get(a, 0) + 1
    return counts


def test_sloppy_failures_attributed_to_model(backend, sloppy_calibrated):
    """Failures injected by model bias (well-calibrated hardware) ⇒ should be attributed mainly to model."""
    from pabench.runners import CALIBRATED_ARM
    counts = _attribute_all(backend, sloppy_calibrated, CALIBRATED_ARM)
    total = sum(counts.values())
    assert total >= 10  # sloppy must produce enough failures, otherwise the test itself is invalid
    assert counts.get(Attribution.MODEL, 0) / total >= 0.8


def test_precise_on_worn_arm_yields_hardware_attributions(backend, precise_worn):
    """Failures injected by hardware drift (good model) ⇒ hardware attribution should be the majority."""
    from pabench.runners import WORN_ARM
    counts = _attribute_all(backend, precise_worn, WORN_ARM)
    total = sum(counts.values())
    assert total >= 5
    assert counts.get(Attribution.HARDWARE, 0) / total >= 0.5


def test_attribution_written_back_with_versioned_reason(backend, sloppy_calibrated):
    """NFR-6 auditable: the attribution conclusion must carry the rule version and the rule that fired."""
    failed = [e for e in sloppy_calibrated if not e.outcome.success]
    ep = failed[0]
    attribute_episode(ep, TH)
    assert ep.outcome.attribution is not None
    assert ep.outcome.attribution_reason.startswith(f"[{TH.version}]")