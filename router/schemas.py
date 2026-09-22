"""Stable data contracts for the Superpowers adaptive routing layer.

See docs/router/spec.md sections 5 (core data contracts) and 6 (classifier)
for the originating specification. Two deliberate deviations from that spec,
both noted where they occur:

  - `TaskEnvelope.previous_failures` is `list[str]`, not `list["FailureEvidence"]`:
    the spec references a `FailureEvidence` type but never defines its shape
    anywhere in the document. A list of pre-rendered evidence strings is the
    simplest structure that satisfies section 13's escalation-package
    requirements without inventing fields the spec never specified.

  - `ClassifierResult` is not named in the spec. Section 5.2 requires "the
    caller MUST NOT need to understand Laya's output structure" -- this is
    the boundary type that satisfies that requirement: classifier.py produces
    it, policy.py consumes it, and nothing downstream touches Laya's raw
    response shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CommandResult:
    """One verification command's outcome (spec section 14)."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    environmental: bool = False  # True if failure looks infra-caused, not code-caused


@dataclass
class VerificationResult:
    """Spec section 5.4 / 14."""

    passed: bool

    build: str = "not_run"
    tests: str = "not_run"
    lint: str = "not_run"
    typecheck: str = "not_run"
    spec_review: str = "not_run"
    quality_review: str = "not_run"

    commands: list[CommandResult] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    retryable: bool = True
    suspected_context_failure: bool = False
    suspected_reasoning_failure: bool = False


@dataclass
class AttemptRecord:
    """Spec section 5.3."""

    attempt: int
    tier: str
    provider: str
    model: str

    started_at: str
    finished_at: str

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None

    status: str = "unknown"
    worker_report: Optional[str] = None
    patch_ref: Optional[str] = None

    verification: Optional[VerificationResult] = None


@dataclass
class TaskEnvelope:
    """Spec section 5.1. Every model dispatch is normalized into this before routing."""

    task_id: str
    role: str
    title: str
    description: str

    plan_text: Optional[str] = None
    task_brief_path: Optional[str] = None
    repository_root: Optional[str] = None

    expected_files: list[str] = field(default_factory=list)
    expected_file_count: Optional[int] = None
    languages: list[str] = field(default_factory=list)

    acceptance_criteria: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)

    previous_attempts: list[AttemptRecord] = field(default_factory=list)
    previous_failures: list[str] = field(default_factory=list)

    requires_web: bool = False
    requires_repo_search: bool = False
    requires_architecture: bool = False
    destructive_operations: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierResult:
    """Normalized classifier output, before policy is applied.

    Every `Classifier` implementation (Laya-backed or deterministic fallback)
    returns this shape. `complexity` is an int in [0,4] (trivial..expert),
    `verification_strength` an int in [0,3] (strong..weak), the four
    probability fields are floats in [0,1].
    """

    task_type: str
    complexity: int
    verification_strength: int
    cross_file_reasoning: float
    architectural_judgment: float
    ambiguity: float
    debugging_reasoning: float
    confidence: float
    classifier_name: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    """Spec section 5.2. The stable contract callers consume; they never need
    to know whether Laya or the deterministic fallback produced it."""

    task_id: str

    tier: str
    minimum_tier: str
    maximum_tier: str

    task_type: str
    complexity: int
    verification_strength: int

    cross_file_reasoning_probability: float
    architectural_judgment_probability: float
    ambiguity_probability: float
    debugging_probability: float

    classifier_confidence: float

    forced_by_policy: bool = False
    policy_reason: Optional[str] = None
    reason_codes: list[str] = field(default_factory=list)
    classifier_raw: dict[str, Any] = field(default_factory=dict)
