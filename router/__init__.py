"""Superpowers adaptive model router.

This is a fork-specific addition (see docs/router/spec.md) -- not part of
upstream obra/superpowers, and per this repo's own AGENTS.md contribution
policy, not intended to be upstreamed there.

Implemented so far:
  - schemas: TaskEnvelope, RouteDecision, ClassifierResult, AttemptRecord, VerificationResult
  - config: routing.yaml loading with env-var interpolation
  - policy: deterministic tier selection with hard floors/ceilings, confidence handling
  - classifier: Laya adapter (optional dependency) + deterministic fallback
  - cli: `route-task --dry-run`

Not yet implemented (see docs/model-routing.md for the roadmap): the model
broker/provider fallback, SDD dispatch integration, verification/escalation
execution, and telemetry. Nothing in this package dispatches a worker yet.
"""
from .classifier import get_classifier
from .policy import choose_tier
from .schemas import (
    AttemptRecord,
    ClassifierResult,
    CommandResult,
    RouteDecision,
    TaskEnvelope,
    VerificationResult,
)

__all__ = [
    "AttemptRecord",
    "ClassifierResult",
    "CommandResult",
    "RouteDecision",
    "TaskEnvelope",
    "VerificationResult",
    "choose_tier",
    "get_classifier",
]
