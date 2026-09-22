from router.schemas import (
    AttemptRecord,
    CommandResult,
    RouteDecision,
    TaskEnvelope,
    VerificationResult,
)


def test_task_envelope_minimal_construction():
    task = TaskEnvelope(task_id="t1", role="implementer", title="x", description="y")
    assert task.expected_files == []
    assert task.metadata == {}
    assert task.previous_attempts == []


def test_verification_result_defaults_not_run():
    v = VerificationResult(passed=False)
    assert v.build == "not_run"
    assert v.commands == []
    assert v.retryable is True


def test_command_result_round_trip():
    c = CommandResult(command="pytest", exit_code=1, stdout="", stderr="boom", duration_ms=120)
    assert c.environmental is False


def test_attempt_record_verification_optional():
    a = AttemptRecord(
        attempt=1, tier="free_strong", provider="openrouter", model="x",
        started_at="t0", finished_at="t1",
    )
    assert a.verification is None


def test_route_decision_defaults():
    d = RouteDecision(
        task_id="t1", tier="free_strong", minimum_tier="local_small", maximum_tier="frontier_reasoning",
        task_type="implementation", complexity=2, verification_strength=1,
        cross_file_reasoning_probability=0.1, architectural_judgment_probability=0.1,
        ambiguity_probability=0.1, debugging_probability=0.1, classifier_confidence=0.9,
    )
    assert d.forced_by_policy is False
    assert d.reason_codes == []
