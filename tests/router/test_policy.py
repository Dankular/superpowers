import pytest

from router.config import ConfidenceConfig, RoutingConfig, SpeculativeDowngradeConfig
from router.errors import PolicyConflictError
from router.policy import choose_tier, tier_rank
from router.schemas import ClassifierResult, TaskEnvelope

DEFAULT_FLOORS = {
    "architecture": "frontier",
    "final_review": "frontier",
    "security_review": "frontier",
    "concurrency_review": "frontier",
    "multi_file_implementation": "free_strong",
    "nontrivial_review": "free_strong",
    "unknown_cause_debugging": "free_strong",
}


def make_config(**overrides) -> RoutingConfig:
    config = RoutingConfig(
        confidence=ConfidenceConfig(low=0.58, high=0.82),
        default_uncertain_tier="free_strong",
        speculative_downgrade=SpeculativeDowngradeConfig(enabled=True, max_levels=1, requires_strong_verification=True),
        floors=dict(DEFAULT_FLOORS),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_task(**overrides) -> TaskEnvelope:
    defaults = dict(task_id="t1", role="implementer", title="x", description="y")
    defaults.update(overrides)
    return TaskEnvelope(**defaults)


def make_classification(**overrides) -> ClassifierResult:
    defaults = dict(
        task_type="implementation", complexity=2, verification_strength=3,
        cross_file_reasoning=0.1, architectural_judgment=0.1, ambiguity=0.1,
        debugging_reasoning=0.1, confidence=0.9, classifier_name="test",
    )
    defaults.update(overrides)
    return ClassifierResult(**defaults)


def test_one_file_complete_spec_with_tests_routes_cheap():
    task = make_task(expected_files=["a.py"])
    classification = make_classification(task_type="mechanical", complexity=0, verification_strength=0)
    decision = choose_tier(task, classification, make_config())
    assert decision.tier in {"local_small", "free_small"}


def test_multi_file_integration_at_least_free_strong():
    task = make_task(expected_files=["a.py", "b.py", "c.py"])
    classification = make_classification(task_type="implementation", complexity=2)
    decision = choose_tier(task, classification, make_config())
    assert tier_rank(decision.tier) >= tier_rank("free_strong")
    assert "MULTI_FILE_FLOOR" in decision.reason_codes


def test_architecture_task_type_alone_forces_frontier_via_floor():
    task = make_task()
    classification = make_classification(task_type="architecture", architectural_judgment=0.2, verification_strength=0)
    decision = choose_tier(task, classification, make_config())
    assert tier_rank(decision.tier) >= tier_rank("frontier")


def test_final_review_metadata_forces_frontier_floor():
    task = make_task(role="reviewer", metadata={"final_review": True})
    classification = make_classification(task_type="review", complexity=1, verification_strength=3)
    decision = choose_tier(task, classification, make_config())
    assert tier_rank(decision.tier) >= tier_rank("frontier")
    assert "FINAL_REVIEW_FLOOR" in decision.reason_codes


def test_strong_verification_speculative_drop_is_at_most_one_level():
    task = make_task(expected_files=["a.py"])
    dropped = make_classification(task_type="implementation", complexity=3, verification_strength=0)
    baseline = make_classification(task_type="implementation", complexity=3, verification_strength=3)
    decision = choose_tier(task, dropped, make_config())
    baseline_decision = choose_tier(task, baseline, make_config())
    assert tier_rank(baseline_decision.tier) - tier_rank(decision.tier) <= 1


def test_weak_verification_no_unsafe_downgrade():
    task = make_task()
    classification = make_classification(task_type="architecture", architectural_judgment=0.9, verification_strength=3)
    decision = choose_tier(task, classification, make_config())
    assert tier_rank(decision.tier) >= tier_rank("frontier")
    assert "STRONG_VERIFICATION_DISCOUNT" not in decision.reason_codes


def test_configured_max_tier_never_exceeded():
    # complexity=4 alone reaches "frontier" via the base proposal, without
    # tripping any structural floor (single file, not architecture/review/debugging),
    # so this isolates the ceiling clamp from floor-vs-ceiling conflict handling.
    task = make_task()
    classification = make_classification(task_type="implementation", complexity=4, architectural_judgment=0.1)
    config = make_config(max_tier="economy")
    decision = choose_tier(task, classification, config)
    assert tier_rank(decision.tier) <= tier_rank("economy")
    assert "CAPPED_BY_MAX_TIER" in decision.reason_codes


def test_configured_floor_never_undercut():
    task = make_task(role="reviewer", metadata={"security_sensitive": True})
    classification = make_classification(task_type="mechanical", complexity=0, verification_strength=0, confidence=0.95)
    decision = choose_tier(task, classification, make_config())
    assert decision.tier == "frontier"


def test_floor_exceeding_ceiling_raises_conflict():
    task = make_task(metadata={"security_sensitive": True})
    classification = make_classification(task_type="mechanical", complexity=0, confidence=0.95)
    config = make_config(max_tier="free_small")
    with pytest.raises(PolicyConflictError):
        choose_tier(task, classification, config)


def test_local_only_forces_local_tier_ceiling():
    task = make_task(expected_files=["a.py"])
    classification = make_classification(task_type="mechanical", complexity=0, confidence=0.95)
    config = make_config(local_only=True, floors={})
    decision = choose_tier(task, classification, config)
    assert decision.tier == "local_small"


def test_low_confidence_does_not_jump_to_frontier_for_ordinary_task():
    task = make_task()
    classification = make_classification(task_type="implementation", complexity=1, confidence=0.2, verification_strength=0)
    decision = choose_tier(task, classification, make_config())
    assert decision.tier != "frontier"


def test_deterministic_fallback_confidence_never_forces_default_uncertain_tier():
    # The fallback classifier always reports confidence=0.3 (below the
    # default low threshold), which must NOT trigger the confidence-policy
    # safety net -- that sentinel means "not ML", not "unsure about this
    # task". A trivial single-file mechanical task should stay cheap even
    # under fallback (unlike the equivalent low-confidence case from a real
    # classifier, tested above).
    task = make_task(expected_files=["a.py"])
    classification = make_classification(
        task_type="mechanical", complexity=0, verification_strength=3,
        confidence=0.3, classifier_name="deterministic_fallback",
    )
    decision = choose_tier(task, classification, make_config())
    assert decision.tier == "local_small"
    assert decision.forced_by_policy is False


def test_low_confidence_with_high_ambiguity_and_weak_verification_prefers_frontier():
    task = make_task()
    classification = make_classification(
        task_type="implementation", complexity=1, confidence=0.2, ambiguity=0.9, verification_strength=3,
    )
    decision = choose_tier(task, classification, make_config())
    assert decision.tier == "frontier"
