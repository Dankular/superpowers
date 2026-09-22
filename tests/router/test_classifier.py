import builtins

import pytest

from router.classifier import (
    DeterministicFallbackClassifier,
    LayaClassifier,
    ROUTING_QUESTIONS,
    get_classifier,
)
from router.config import RoutingConfig
from router.errors import ClassifierUnavailableError
from router.schemas import TaskEnvelope


def make_task(**overrides) -> TaskEnvelope:
    defaults = dict(task_id="t1", role="implementer", title="Fix bug", description="Fix the off-by-one error")
    defaults.update(overrides)
    return TaskEnvelope(**defaults)


class FakeLayaRouter:
    """Stands in for laya.Router; returns the exact response shape produced
    by laya 0.3.5's `Agent.system_one` (verified against the real source)."""

    def __init__(self, answers):
        self._answers = answers
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append((state, questions, model))
        assert set(questions) == set(ROUTING_QUESTIONS)
        return {"model": "laya-rl-agent", "answers": self._answers, "usage": {"input_tokens": 10, "output_tokens": 0}}


def _fake_answers(**overrides):
    base = {
        "task_type": {"type": "choice", "choice": "implementation", "confidence": 0.9},
        "complexity": {"type": "score", "score": 2.3, "confidence": 0.8},
        "verification_strength": {"type": "score", "score": 1.1, "confidence": 0.85},
        "cross_file_reasoning": {"type": "noul", "noul": 0.4, "confidence": 0.7},
        "architectural_judgment": {"type": "noul", "noul": 0.1, "confidence": 0.9},
        "ambiguity": {"type": "noul", "noul": 0.2, "confidence": 0.9},
        "debugging_reasoning": {"type": "noul", "noul": 0.15, "confidence": 0.9},
    }
    base.update(overrides)
    return base


def test_laya_classifier_normalizes_real_response_shape():
    fake_router = FakeLayaRouter(_fake_answers())
    classifier = LayaClassifier(router_factory=lambda: fake_router)
    result = classifier.classify(make_task())
    assert result.task_type == "implementation"
    assert result.complexity == 2  # round(2.3)
    assert result.verification_strength == 1  # round(1.1)
    assert result.cross_file_reasoning == 0.4
    assert result.confidence == pytest.approx(0.7)  # min of per-question confidences
    assert fake_router.calls[0][2] == "english"


def test_laya_classifier_excludes_metadata_from_state():
    fake_router = FakeLayaRouter(_fake_answers())
    classifier = LayaClassifier(router_factory=lambda: fake_router)
    classifier.classify(make_task(metadata={"parent_transcript": "SECRET" * 100}))
    state = fake_router.calls[0][0]
    assert "SECRET" not in state


def test_laya_unavailable_raises_when_import_fails(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "laya":
            raise ImportError("no laya installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    classifier = LayaClassifier()
    with pytest.raises(ClassifierUnavailableError):
        classifier.classify(make_task())


def test_laya_unavailable_when_factory_raises():
    def broken_factory():
        raise RuntimeError("network unreachable")

    classifier = LayaClassifier(router_factory=broken_factory)
    with pytest.raises(ClassifierUnavailableError):
        classifier.classify(make_task())


def test_deterministic_fallback_always_low_confidence():
    classifier = DeterministicFallbackClassifier()
    result = classifier.classify(make_task())
    assert result.confidence < 0.58  # below the default low-confidence threshold, by design


def test_deterministic_fallback_mechanical_for_small_single_file_task():
    classifier = DeterministicFallbackClassifier()
    result = classifier.classify(make_task(description="short", expected_files=["a.py"]))
    assert result.task_type == "mechanical"


def test_deterministic_fallback_architecture_for_flagged_task():
    classifier = DeterministicFallbackClassifier()
    result = classifier.classify(make_task(requires_architecture=True))
    assert result.task_type == "architecture"


def test_get_classifier_falls_back_when_laya_not_selected():
    config = RoutingConfig(classifier="deterministic")
    classifier = get_classifier(config)
    assert isinstance(classifier, DeterministicFallbackClassifier)


def test_get_classifier_falls_back_when_laya_router_raises(capsys):
    config = RoutingConfig(classifier="laya")

    def broken_factory():
        raise RuntimeError("network unreachable")

    classifier = get_classifier(config, router_factory=broken_factory)
    assert isinstance(classifier, DeterministicFallbackClassifier)
    assert "falling back to deterministic policy" in capsys.readouterr().err


def test_get_classifier_uses_laya_when_available():
    fake_router = FakeLayaRouter(_fake_answers())
    config = RoutingConfig(classifier="laya")
    classifier = get_classifier(config, router_factory=lambda: fake_router)
    assert isinstance(classifier, LayaClassifier)
