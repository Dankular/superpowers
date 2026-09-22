"""Deterministic routing policy.

The classifier (Laya or the deterministic fallback) proposes a starting
tier; this module is the inspectable, testable layer that turns that
proposal into a final RouteDecision, enforcing hard floors and ceilings.
Laya probabilities are never treated as ground truth here -- see
`_apply_confidence_policy`.

See docs/router/spec.md sections 7-10 for the originating design.
"""
from __future__ import annotations

from typing import Optional

from .config import RoutingConfig
from .errors import PolicyConflictError, RouterConfigError
from .schemas import ClassifierResult, RouteDecision, TaskEnvelope

TIERS = ["local_small", "free_small", "free_strong", "economy", "frontier", "frontier_reasoning"]


def tier_rank(tier: str) -> int:
    try:
        return TIERS.index(tier)
    except ValueError:
        raise RouterConfigError(f"unknown tier {tier!r}; expected one of {TIERS}")


def one_tier_lower(tier: str) -> str:
    return TIERS[max(0, tier_rank(tier) - 1)]


def one_tier_higher(tier: str) -> str:
    return TIERS[min(len(TIERS) - 1, tier_rank(tier) + 1)]


def max_tier(a: str, b: str) -> str:
    return a if tier_rank(a) >= tier_rank(b) else b


def min_tier(a: str, b: str) -> str:
    return a if tier_rank(a) <= tier_rank(b) else b


class RoutingFeatures:
    """Deterministic-policy input, distilled from a ClassifierResult + TaskEnvelope."""

    def __init__(
        self,
        task_type: str,
        complexity: int,
        verification_strength: int,
        cross_file_reasoning: float,
        architectural_judgment: float,
        ambiguity: float,
        debugging_reasoning: float,
        classifier_confidence: float,
        expected_file_count: int,
        is_final_review: bool = False,
        is_security_sensitive: bool = False,
        is_concurrency_sensitive: bool = False,
    ):
        self.task_type = task_type
        self.complexity = complexity
        self.verification_strength = verification_strength
        self.cross_file_reasoning = cross_file_reasoning
        self.architectural_judgment = architectural_judgment
        self.ambiguity = ambiguity
        self.debugging_reasoning = debugging_reasoning
        self.classifier_confidence = classifier_confidence
        self.expected_file_count = expected_file_count
        self.is_final_review = is_final_review
        self.is_security_sensitive = is_security_sensitive
        self.is_concurrency_sensitive = is_concurrency_sensitive


def features_from(task: TaskEnvelope, classification: ClassifierResult) -> RoutingFeatures:
    file_count = task.expected_file_count
    if file_count is None:
        file_count = len(task.expected_files)
    return RoutingFeatures(
        task_type=classification.task_type,
        complexity=classification.complexity,
        verification_strength=classification.verification_strength,
        cross_file_reasoning=classification.cross_file_reasoning,
        architectural_judgment=classification.architectural_judgment,
        ambiguity=classification.ambiguity,
        debugging_reasoning=classification.debugging_reasoning,
        classifier_confidence=classification.confidence,
        expected_file_count=file_count,
        is_final_review=bool(task.metadata.get("final_review", False)),
        is_security_sensitive=bool(task.metadata.get("security_sensitive", False)),
        is_concurrency_sensitive=bool(task.metadata.get("concurrency_sensitive", False)),
    )


def structural_floor(features: RoutingFeatures, floors: dict[str, str]) -> tuple[str, list[str]]:
    """Hard floors (spec section 9.1), keyed by explicit/structural signals only.

    Deliberately does NOT keyword-sniff task text for "security" or
    "concurrency" -- those floors require an explicit TaskEnvelope.metadata
    flag (`security_sensitive`, `concurrency_sensitive`). Guessing sensitivity
    from free text is exactly the kind of unreliable inference this router is
    supposed to avoid: an undetected security-sensitive task silently landing
    on a cheap worker is worse than requiring callers to say so explicitly.
    """
    floor = TIERS[0]
    codes: list[str] = []

    def bump(floor_key: str, default_tier: str, code: str) -> None:
        nonlocal floor
        floor = max_tier(floor, floors.get(floor_key, default_tier))
        codes.append(code)

    if features.task_type == "architecture" or features.architectural_judgment >= 0.70:
        bump("architecture", "frontier", "ARCHITECTURE_FLOOR")
    if features.is_final_review:
        bump("final_review", "frontier", "FINAL_REVIEW_FLOOR")
    if features.is_security_sensitive:
        bump("security_review", "frontier", "SECURITY_REVIEW_FLOOR")
    if features.is_concurrency_sensitive:
        bump("concurrency_review", "frontier", "CONCURRENCY_FLOOR")
    if features.task_type in {"implementation", "debugging", "refactor"} and features.expected_file_count > 1:
        bump("multi_file_implementation", "free_strong", "MULTI_FILE_FLOOR")
    if features.task_type == "review" and (features.complexity >= 3 or features.ambiguity >= 0.6):
        bump("nontrivial_review", "free_strong", "NONTRIVIAL_REVIEW_FLOOR")
    if features.task_type == "debugging" and features.debugging_reasoning >= 0.6 and features.ambiguity >= 0.5:
        bump("unknown_cause_debugging", "free_strong", "UNKNOWN_CAUSE_DEBUGGING_FLOOR")

    return floor, codes


def _base_proposal(features: RoutingFeatures, allow_speculative_downgrade: bool) -> tuple[str, list[str]]:
    """Spec section 9.3's deterministic first-cut policy."""
    codes: list[str] = []

    if features.architectural_judgment >= 0.70:
        proposed = "frontier"
        codes.append("HIGH_ARCHITECTURAL_JUDGMENT")
    elif features.task_type == "architecture":
        proposed = "frontier"
        codes.append("ARCHITECTURE_TASK_TYPE")
    elif features.complexity >= 4:
        proposed = "frontier"
        codes.append("EXPERT_COMPLEXITY")
    elif features.complexity >= 3:
        proposed = "free_strong"
        codes.append("DIFFICULT_COMPLEXITY")
    elif features.cross_file_reasoning >= 0.65:
        proposed = "free_strong"
        codes.append("CROSS_FILE_REASONING")
    elif features.task_type == "mechanical":
        proposed = "local_small"
        codes.append("MECHANICAL_TASK")
    else:
        proposed = "free_small"
        codes.append("ROUTINE_TASK")

    # Strong objective verification (score index 0 == "strong" in the
    # verification_strength question's criteria list) permits one speculative
    # tier downgrade -- a risk reducer, not proof a weak model will succeed
    # (spec section 9.2).
    if allow_speculative_downgrade and features.verification_strength == 0:
        proposed = one_tier_lower(proposed)
        codes.append("STRONG_VERIFICATION_DISCOUNT")

    # High ambiguity with real verification coverage: don't let a cheap model
    # guess at underspecified requirements just because tests exist to catch
    # a wrong guess after the fact.
    if features.ambiguity >= 0.80 and features.verification_strength >= 2:
        proposed = one_tier_higher(proposed)
        codes.append("AMBIGUITY_UPGRADE")

    return proposed, codes


def _apply_confidence_policy(
    proposed: str, features: RoutingFeatures, config: RoutingConfig
) -> tuple[str, bool, list[str]]:
    """Spec section 10: low classifier confidence must not itself justify
    jumping to the most expensive tier, nor license a cheap model to guess.

    Section 10 states qualitative rules (1-4; rule 5 is calibration logging,
    which belongs to telemetry and isn't implemented in this phase) rather
    than an algorithm. This is a deterministic, documented translation of
    those rules -- not a verbatim spec algorithm, since none was given.
    """
    if features.classifier_confidence >= config.confidence.low:
        return proposed, False, []

    codes = ["LOW_CLASSIFIER_CONFIDENCE"]
    if features.architectural_judgment >= 0.70 or features.task_type == "architecture":
        codes.append("LOW_CONFIDENCE_ARCHITECTURE_SAFETY")
        return "frontier", True, codes
    if features.ambiguity >= 0.80 and features.verification_strength >= 2:
        codes.append("LOW_CONFIDENCE_AMBIGUITY_SAFETY")
        return "frontier", True, codes
    codes.append("LOW_CONFIDENCE_DEFAULT_UNCERTAIN_TIER")
    return config.default_uncertain_tier, True, codes


def choose_tier(task: TaskEnvelope, classification: ClassifierResult, config: RoutingConfig) -> RouteDecision:
    features = features_from(task, classification)

    floor, floor_codes = structural_floor(features, config.floors)
    proposed, proposal_codes = _base_proposal(features, config.speculative_downgrade.enabled)

    if classification.classifier_name == "deterministic_fallback":
        # The fallback's fixed confidence=0.3 is a marker meaning "this is a
        # structural heuristic, not ML", not a per-task uncertainty signal --
        # the confidence-policy safety net exists for a real classifier that
        # might be unsure about a *specific* task. Applying it here would
        # unconditionally override every fallback decision (including
        # obviously-mechanical ones) with default_uncertain_tier, which
        # directly contradicts the fallback's own task-type-based routing.
        confidence_forced, confidence_codes = False, []
    else:
        proposed, confidence_forced, confidence_codes = _apply_confidence_policy(proposed, features, config)

    final = max_tier(proposed, floor)
    forced_by_policy = confidence_forced or tier_rank(floor) > tier_rank(proposed)

    ceiling: Optional[str] = config.max_tier
    if config.local_only:
        ceiling = "local_small" if ceiling is None else min_tier(ceiling, "local_small")

    if ceiling is not None:
        if tier_rank(floor) > tier_rank(ceiling):
            raise PolicyConflictError(
                f"task {task.task_id!r} requires floor {floor!r} but the configured "
                f"ceiling is {ceiling!r} (max_tier/local_only); refusing to silently pick a side"
            )
        if tier_rank(final) > tier_rank(ceiling):
            final = ceiling
            forced_by_policy = True
            confidence_codes.append("CAPPED_BY_MAX_TIER")

    reason_codes = floor_codes + proposal_codes + confidence_codes
    policy_reason = "; ".join(reason_codes) if forced_by_policy else None

    return RouteDecision(
        task_id=task.task_id,
        tier=final,
        minimum_tier=floor,
        maximum_tier=ceiling or TIERS[-1],
        task_type=classification.task_type,
        complexity=classification.complexity,
        verification_strength=classification.verification_strength,
        cross_file_reasoning_probability=classification.cross_file_reasoning,
        architectural_judgment_probability=classification.architectural_judgment,
        ambiguity_probability=classification.ambiguity,
        debugging_probability=classification.debugging_reasoning,
        classifier_confidence=classification.confidence,
        forced_by_policy=forced_by_policy,
        policy_reason=policy_reason,
        reason_codes=reason_codes,
        classifier_raw=classification.raw,
    )
