import json
import textwrap

from router.cli import main


def test_dry_run_end_to_end(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "task_id": "task-1",
        "role": "implementer",
        "title": "Add a getter",
        "description": "Add a trivial getter method to one file.",
        "expected_files": ["src/foo.py"],
    }))
    config_path = tmp_path / "routing.yaml"
    config_path.write_text(textwrap.dedent("""
        routing:
          enabled: true
          classifier: deterministic
          default_tier: free_strong
          confidence:
            low: 0.58
            high: 0.82
            default_uncertain_tier: free_strong
          floors: {}
    """))

    exit_code = main(["--input", str(task_path), "--config", str(config_path), "--dry-run"])
    assert exit_code == 0

    out = json.loads(capsys.readouterr().out)
    assert out["task_id"] == "task-1"
    assert out["tier"] in {"local_small", "free_small", "free_strong", "economy", "frontier", "frontier_reasoning"}
    assert out["classifier"]["name"] == "deterministic_fallback"


def test_non_dry_run_refused(capsys):
    exit_code = main(["--input", "whatever.json"])
    assert exit_code == 2
    assert "broker" in capsys.readouterr().err


def test_unknown_task_field_rejected(tmp_path, capsys):
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({
        "task_id": "task-1", "role": "implementer", "title": "x", "description": "y",
        "not_a_real_field": True,
    }))
    exit_code = main(["--input", str(task_path), "--dry-run"])
    assert exit_code == 1
    assert "unknown fields" in capsys.readouterr().err
