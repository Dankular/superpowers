import textwrap

import pytest

from router.config import load_routing_config, load_yaml
from router.errors import RouterConfigError


def test_env_var_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MODEL", "gpt-real")
    monkeypatch.delenv("MISSING_ROUTER_TEST_VAR", raising=False)
    p = tmp_path / "c.yaml"
    p.write_text("model: ${MY_MODEL}\nfallback: ${MISSING_ROUTER_TEST_VAR:-default-value}\n")
    data = load_yaml(p)
    assert data["model"] == "gpt-real"
    assert data["fallback"] == "default-value"


def test_load_routing_config_reads_all_fields(tmp_path):
    p = tmp_path / "routing.yaml"
    p.write_text(textwrap.dedent("""
        routing:
          enabled: true
          classifier: laya
          default_tier: free_strong
          confidence:
            low: 0.5
            high: 0.9
            default_uncertain_tier: economy
          floors:
            architecture: frontier
        repository_policy:
          max_tier: frontier
          local_only: false
    """))
    cfg = load_routing_config(p)
    assert cfg.classifier == "laya"
    assert cfg.confidence.low == 0.5
    assert cfg.confidence.high == 0.9
    assert cfg.default_uncertain_tier == "economy"
    assert cfg.floors["architecture"] == "frontier"
    assert cfg.max_tier == "frontier"
    assert cfg.local_only is False


def test_load_routing_config_defaults_when_sections_missing(tmp_path):
    p = tmp_path / "routing.yaml"
    p.write_text("routing:\n  enabled: true\n")
    cfg = load_routing_config(p)
    assert cfg.classifier == "laya"
    assert cfg.default_tier == "free_strong"
    assert cfg.confidence.low == 0.58
    assert cfg.max_tier is None
    assert cfg.local_only is False


def test_load_yaml_missing_file_raises(tmp_path):
    with pytest.raises(RouterConfigError):
        load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_non_mapping_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(RouterConfigError):
        load_yaml(p)
