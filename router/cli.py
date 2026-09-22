"""`route-task` CLI.

Only --dry-run is implemented in this phase -- concrete provider/model
selection requires the model broker, which does not exist yet. Non-dry-run
invocation is refused with a clear error rather than silently returning a
decision with no way to act on it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .classifier import get_classifier
from .config import load_routing_config
from .errors import RouterError
from .policy import choose_tier
from .schemas import TaskEnvelope

DEFAULT_ROUTING_CONFIG = Path(__file__).resolve().parent.parent / "config" / "routing.example.yaml"

_TASK_FIELDS = {f.name for f in dataclasses.fields(TaskEnvelope)}


def _load_task(path: str) -> TaskEnvelope:
    with open(path) as f:
        data = json.load(f)
    unknown = set(data) - _TASK_FIELDS
    if unknown:
        raise RouterError(f"task envelope {path!r} has unknown fields: {sorted(unknown)}")
    return TaskEnvelope(**data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="route-task")
    parser.add_argument("--input", required=True, help="path to a TaskEnvelope JSON file")
    parser.add_argument("--config", default=str(DEFAULT_ROUTING_CONFIG), help="path to routing.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the routing decision only")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dry_run:
        print(
            "route-task only supports --dry-run in this phase: concrete provider/model "
            "selection needs the model broker, which is not built yet. "
            "See docs/model-routing.md.",
            file=sys.stderr,
        )
        return 2

    try:
        task = _load_task(args.input)
        config = load_routing_config(args.config)
        classifier = get_classifier(config)
        classification = classifier.classify(task)
        decision = choose_tier(task, classification, config)
    except RouterError as exc:
        print(f"route-task: {exc}", file=sys.stderr)
        return 1

    print(f"Task: {task.title}", file=sys.stderr)
    print(f"Classifier: {classifier.name}", file=sys.stderr)
    print(
        f"  type={decision.task_type} complexity={decision.complexity} "
        f"verification_strength={decision.verification_strength} "
        f"confidence={decision.classifier_confidence:.2f}",
        file=sys.stderr,
    )
    print(
        f"Selected tier: {decision.tier} (floor={decision.minimum_tier}, ceiling={decision.maximum_tier})",
        file=sys.stderr,
    )
    if decision.policy_reason:
        print(f"Policy: {decision.policy_reason}", file=sys.stderr)

    print(
        json.dumps(
            {
                "task_id": decision.task_id,
                "tier": decision.tier,
                "minimum_tier": decision.minimum_tier,
                "maximum_tier": decision.maximum_tier,
                "classifier": {
                    "name": classifier.name,
                    "task_type": decision.task_type,
                    "complexity": decision.complexity,
                    "verification_strength": decision.verification_strength,
                    "confidence": decision.classifier_confidence,
                },
                "forced_by_policy": decision.forced_by_policy,
                "reason_codes": decision.reason_codes,
                "policy_reason": decision.policy_reason,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
