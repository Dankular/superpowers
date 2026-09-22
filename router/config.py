"""YAML config loading for the routing layer, with env-var interpolation.

Supports `${VAR}` and `${VAR:-default}` in string values, matching the
syntax used throughout config/routing.example.yaml and config/models.example.yaml.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .errors import RouterConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            var, _, default = m.groups()
            if var in os.environ:
                return os.environ[var]
            return default if default is not None else ""
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_yaml(path: Union[str, Path]) -> dict:
    p = Path(path)
    if not p.exists():
        raise RouterConfigError(f"config file not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RouterConfigError(f"config file {p} must contain a YAML mapping at the top level")
    return _interpolate(raw)


@dataclass
class ConfidenceConfig:
    low: float = 0.58
    high: float = 0.82


@dataclass
class SpeculativeDowngradeConfig:
    enabled: bool = True
    max_levels: int = 1
    requires_strong_verification: bool = True


@dataclass
class RoutingConfig:
    """Everything policy.py needs. Deliberately excludes provider/broker
    fields (config/models.example.yaml) -- concrete provider/model selection
    is a later phase (see docs/router/spec.md section 11) and isn't consumed
    by any code yet.
    """

    enabled: bool = True
    classifier: str = "laya"

    default_tier: str = "free_strong"
    default_uncertain_tier: str = "free_strong"
    final_review_min_tier: str = "frontier"

    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    speculative_downgrade: SpeculativeDowngradeConfig = field(default_factory=SpeculativeDowngradeConfig)

    floors: dict[str, str] = field(default_factory=dict)

    max_tier: Optional[str] = None
    local_only: bool = False

    raw: dict[str, Any] = field(default_factory=dict)


def load_routing_config(path: Union[str, Path]) -> RoutingConfig:
    data = load_yaml(path)
    r = data.get("routing", {})
    conf = r.get("confidence", {})
    spec_down = r.get("speculative_downgrade", {})
    repo_policy = data.get("repository_policy", {})

    return RoutingConfig(
        enabled=r.get("enabled", True),
        classifier=r.get("classifier", "laya"),
        default_tier=r.get("default_tier", "free_strong"),
        default_uncertain_tier=conf.get("default_uncertain_tier", r.get("default_tier", "free_strong")),
        final_review_min_tier=r.get("final_review_min_tier", "frontier"),
        confidence=ConfidenceConfig(
            low=conf.get("low", 0.58),
            high=conf.get("high", 0.82),
        ),
        speculative_downgrade=SpeculativeDowngradeConfig(
            enabled=spec_down.get("enabled", True),
            max_levels=spec_down.get("max_levels", 1),
            requires_strong_verification=spec_down.get("requires_strong_verification", True),
        ),
        floors=dict(r.get("floors", {})),
        max_tier=repo_policy.get("max_tier"),
        local_only=repo_policy.get("local_only", False),
        raw=data,
    )
