"""Classifier adapters: normalize any classifier's output into ClassifierResult.

Two implementations ship in this phase:
  - LayaClassifier: wraps the real `laya` package (optional dependency).
  - DeterministicFallbackClassifier: no ML, always available, used when Laya
    is disabled, not installed, or fails to load.

VERIFIED vs ASSUMED, checked directly against laya 0.3.5's source (the wheel
on PyPI, not its README or any secondhand description):

  VERIFIED: `Agent.system_one` / `Router.predict(state, questions, model=...)`
  mechanically accepts an arbitrary custom `questions` dict of type
  choice/score/noul (exactly ROUTING_QUESTIONS below) and returns a
  probability + calibrated-confidence answer per question. The response
  shape used in `_normalize` below (answers[qid]["choice"|"score"|"noul"],
  answers[qid]["confidence"]) is copied from `Agent.system_one`'s actual
  return statement, not guessed.

  NOT VERIFIED, and not knowable without running an eval: whether the
  shipped checkpoints ("english" is Laya's general-purpose default; there is
  also "multilingual" and a "typed-decisions" checkpoint fine-tuned on four
  fixed, unrelated workflows -- agent-trace observability, customer service,
  invoice processing, security incidents) produce *meaningful* judgments for
  this out-of-domain use. Laya's own benchmark data covers customer-support
  intent classification, XNLI, and those four fixed workflows -- nothing
  about source code or software-engineering task classification. Treat
  LayaClassifier's outputs as unvalidated until telemetry (a later phase)
  demonstrates they correlate with actual task difficulty. The deterministic
  fallback and the confidence-policy safety rails in policy.py exist
  precisely because this is unverified.
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Optional, Protocol

from .config import RoutingConfig
from .errors import ClassifierUnavailableError
from .schemas import ClassifierResult, TaskEnvelope

# Field names and types match laya's real question schema exactly (type:
# choice/score/noul; criteria as a dict for choice, an ordered list for
# score; no criteria for noul) -- confirmed against laya/agent.py's
# `Agent._to_internal`, which is what actually parses these.
ROUTING_QUESTIONS: dict[str, dict[str, Any]] = {
    "task_type": {
        "type": "choice",
        "instructions": "Classify the primary software-engineering task described in `state`.",
        "criteria": {
            "mechanical": "A deterministic or repetitive edit with little judgment.",
            "implementation": "Implement specified functionality.",
            "debugging": "Find and repair incorrect behaviour.",
            "refactor": "Restructure implementation while preserving behaviour.",
            "review": "Review an existing change or proposed implementation.",
            "architecture": "Make broad design, interface, or architectural decisions.",
            "research": "Discover or reconcile unfamiliar technical information.",
            "planning": "Break requirements into an implementation plan.",
        },
    },
    "complexity": {
        "type": "score",
        "instructions": "Estimate reasoning complexity required for correct completion of `state`.",
        "criteria": ["trivial", "routine", "moderate", "difficult", "expert"],
    },
    "verification_strength": {
        "type": "score",
        "instructions": "How objectively can success be verified after the worker finishes `state`?",
        "criteria": [
            "strong: deterministic tests/build/checks cover the requested behaviour",
            "good: most important behaviour can be checked automatically",
            "partial: only some requirements can be checked automatically",
            "weak: correctness relies substantially on judgment or review",
        ],
    },
    "cross_file_reasoning": {
        "type": "noul",
        "instructions": (
            "Does correct completion of `state` require coordinating behaviour "
            "across multiple files, modules, services, or APIs?"
        ),
    },
    "architectural_judgment": {
        "type": "noul",
        "instructions": "Does `state` require architecture, API design, trade-off analysis, or broad system judgment?",
    },
    "ambiguity": {
        "type": "noul",
        "instructions": "Are the requirements in `state` materially ambiguous or underspecified?",
    },
    "debugging_reasoning": {
        "type": "noul",
        "instructions": "Does `state` require diagnosing a cause that is not already known?",
    },
}

_COMPLEXITY_LEVELS = len(ROUTING_QUESTIONS["complexity"]["criteria"])  # 5 -> scores land in [0,4]
_VERIFICATION_LEVELS = len(ROUTING_QUESTIONS["verification_strength"]["criteria"])  # 4 -> [0,3]


def _task_text(task: TaskEnvelope) -> str:
    """Build the text `state` handed to the classifier.

    Deliberately excludes TaskEnvelope.metadata (which is where a caller
    might stash a parent-session transcript) and any repository source: only
    the task's own specification and structural summary go to the
    classifier, per the requirement that the router must not automatically
    dump the entire parent transcript into Laya.
    """
    parts = [f"Role: {task.role}", f"Title: {task.title}", "", task.description]
    if task.acceptance_criteria:
        parts.append("\nAcceptance criteria:\n" + "\n".join(f"- {c}" for c in task.acceptance_criteria))
    if task.expected_files:
        parts.append("\nExpected files:\n" + "\n".join(f"- {f}" for f in task.expected_files))
    if task.previous_failures:
        parts.append("\nPrevious failure evidence:\n" + "\n".join(f"- {f}" for f in task.previous_failures))
    return "\n".join(parts)


class Classifier(Protocol):
    name: str

    def classify(self, task: TaskEnvelope) -> ClassifierResult: ...


class LayaClassifier:
    """Wraps `laya.Router` behind the stable `Classifier` interface.

    `router_factory` exists purely for testing: it lets tests inject a fake
    with the same `.predict(state, questions, model=...)` shape as the real
    `laya.Router`, so unit tests never download real model weights.
    """

    name = "laya"

    def __init__(self, router_factory: Optional[Callable[[], Any]] = None, model: str = "english"):
        self._router_factory = router_factory
        self._router: Optional[Any] = None
        # Explicit model: our custom questions never match any of laya's four
        # fixed typed-decisions workflows (exact question-id-set match, per
        # laya/router.py's `match_typed_decisions_workflow`), so its
        # auto-task-detection would never fire for us anyway. "english" is
        # laya's documented general-purpose default checkpoint.
        self._model = model

    def _get_router(self) -> Any:
        if self._router is not None:
            return self._router
        if self._router_factory is not None:
            try:
                self._router = self._router_factory()
            except Exception as exc:
                raise ClassifierUnavailableError(f"router_factory failed: {exc}") from exc
            return self._router
        try:
            from laya import Router
        except ImportError as exc:
            raise ClassifierUnavailableError(
                "laya is not installed; install the 'routing' extra "
                "(pip install '.[routing]') or set routing.classifier to something "
                "other than 'laya' in routing.yaml"
            ) from exc
        try:
            self._router = Router(preload=True, auto_task_detection=False)
        except Exception as exc:  # model download/load failure, no network, OOM, etc.
            raise ClassifierUnavailableError(f"laya.Router failed to load: {exc}") from exc
        return self._router

    def classify(self, task: TaskEnvelope) -> ClassifierResult:
        router = self._get_router()
        state = _task_text(task)
        try:
            result = router.predict(state, ROUTING_QUESTIONS, model=self._model)
        except Exception as exc:
            raise ClassifierUnavailableError(f"laya prediction failed: {exc}") from exc
        return self._normalize(result)

    @staticmethod
    def _normalize(result: dict[str, Any]) -> ClassifierResult:
        answers = result["answers"]
        confidences = [a["confidence"] for a in answers.values()]
        # Laya doesn't specify how to combine seven independent per-question
        # confidences into one classifier_confidence scalar; taking the
        # minimum is a deliberate, conservative choice documented here rather
        # than a value the spec or the package prescribes: the classifier is
        # only as trustworthy as its least-confident answer.
        overall_confidence = min(confidences) if confidences else 0.0

        complexity_score = answers["complexity"]["score"]
        verification_score = answers["verification_strength"]["score"]

        return ClassifierResult(
            task_type=answers["task_type"]["choice"],
            complexity=round(max(0, min(_COMPLEXITY_LEVELS - 1, complexity_score))),
            verification_strength=round(max(0, min(_VERIFICATION_LEVELS - 1, verification_score))),
            cross_file_reasoning=answers["cross_file_reasoning"]["noul"],
            architectural_judgment=answers["architectural_judgment"]["noul"],
            ambiguity=answers["ambiguity"]["noul"],
            debugging_reasoning=answers["debugging_reasoning"]["noul"],
            confidence=overall_confidence,
            classifier_name="laya",
            raw=result,
        )


class DeterministicFallbackClassifier:
    """Structural fallback used when Laya is disabled, uninstalled, or fails
    to load. No ML; always available; never claims high confidence.

    This is a safety net, not a substitute classifier -- it exists so the
    router remains usable when Laya can't run, not to approximate its
    judgment.
    """

    name = "deterministic_fallback"

    def classify(self, task: TaskEnvelope) -> ClassifierResult:
        file_count = task.expected_file_count if task.expected_file_count is not None else len(task.expected_files)
        role = (task.role or "").lower()
        title_desc = f"{task.title} {task.description}".lower()

        if task.requires_architecture or role in {"architect", "final_reviewer"}:
            task_type = "architecture"
            complexity = 4
        elif role in {"reviewer", "task_reviewer", "code_reviewer"}:
            task_type = "review"
            complexity = 2
        elif "debug" in role or "debug" in title_desc:
            task_type = "debugging"
            complexity = 2
        elif file_count <= 1 and len(task.description) < 400:
            task_type = "mechanical"
            complexity = 0
        else:
            task_type = "implementation"
            complexity = 2

        verification_strength = 0 if task.verification_commands else 3

        return ClassifierResult(
            task_type=task_type,
            complexity=complexity,
            verification_strength=verification_strength,
            cross_file_reasoning=1.0 if file_count > 1 else 0.0,
            architectural_judgment=1.0 if task_type == "architecture" else 0.0,
            ambiguity=0.5,  # unknown; the fallback never claims to know this
            debugging_reasoning=1.0 if task_type == "debugging" else 0.0,
            confidence=0.3,  # always low -- forces conservative confidence-policy handling
            classifier_name="deterministic_fallback",
            raw={},
        )


def get_classifier(config: RoutingConfig, router_factory: Optional[Callable[[], Any]] = None) -> Classifier:
    """Router failure mode: log and fall back rather than fail closed."""
    if config.classifier == "laya":
        classifier = LayaClassifier(router_factory=router_factory)
        try:
            classifier._get_router()  # eager: fail now, not on the first real task
            return classifier
        except ClassifierUnavailableError as exc:
            print(
                f"[router] laya classifier unavailable, falling back to deterministic policy: {exc}",
                file=sys.stderr,
            )
            return DeterministicFallbackClassifier()
    return DeterministicFallbackClassifier()
