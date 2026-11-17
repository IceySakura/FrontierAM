import hashlib
import json
from pathlib import Path
import runpy

import pytest
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import yaml

ROOT = Path(__file__).resolve().parents[6]
VERSION = ROOT / "tasks/pdhcg-qp/versions/v1"


def test_harbor_package_contains_only_verifier_assets_and_pinned_configuration(tmp_path):
    prepare = runpy.run_path(str(ROOT / "scripts/seal/prepare_pdhcg_qp.py"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"cases": []}')
    destination = tmp_path / "harbor"
    prepare["package"](destination, data, "registry.example/qp:v1")
    config = tomllib.loads((destination / "task.toml").read_text())
    assert config["environment"]["docker_image"] == "registry.example/qp:v1"
    assert config["verifier"]["environment"]["docker_image"] == "registry.example/qp:v1"
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["verifier"]["environment"]["network_mode"] == "no-network"
    assert (destination / "tests/test.sh").is_file()
    assert (destination / "environment").is_dir()
    assert (destination / "tests/reference/qpbench/model.py").is_file()
    assert not (destination / "tests/tests").exists()
    assert not list(destination.rglob("run_local.py"))
    assert (destination / "instruction.md").read_text() == (VERSION / "statement.md").read_text()
    launch = yaml.safe_load((ROOT / "scripts/rounds/pdhcg-qp-harbor.yaml").read_text())
    assert launch["task_name"] == "harbor_trial_task"
    assert launch["agent_config"]["version"] == "0.114.0"


def test_task_registration_and_local_baseline_mapping():
    from arena.tasks import TASKS
    task = TASKS["pdhcg-qp-v1"]
    metrics = {"raw": {"exit_status": "finished"}, "split": "private",
               "summary": {"valid": True, "p": 1.0, "g": 2.0, "build": {"outcome": "completed"}}}
    assert task.baseline_ok(metrics)[0]
    assert task.score({}, metrics, {"metrics": metrics})["score"] == 50.0
    other = {**metrics, "split": "public"}
    assert task.score({}, metrics, {"metrics": other})["score"] is None
    broken = {**metrics, "raw": {"exit_status": "infrastructure_failure"}}
    assert task.score({}, broken, None)["status"] == "unmeasured"


def test_grader_does_not_write_reward_for_mismatched_calibration(tmp_path):
    from grade import main
    path = tmp_path / "input.npz"
    path.write_bytes(b"trusted input not parsed before checking the anchor")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"id": "a", "path": "input.npz",
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    anchor = tmp_path / "calibration.json"
    anchor.write_text(json.dumps({"status": "calibrated", "manifest_sha256": "wrong-split"}))
    reward = tmp_path / "reward.json"
    reward.write_text('{"reward": 1}')
    with pytest.raises(ValueError, match="mismatched"):
        main(["--manifest", str(manifest), "--calibration", str(anchor), "--reward", str(reward)])
    assert not reward.exists()


def test_build_failure_produces_numeric_harbor_reward(tmp_path):
    from grade import main
    source = tmp_path / "source"
    source.mkdir()  # No run.sh: a candidate failure before any worker is launched.
    path = tmp_path / "input.npz"
    path.write_bytes(b"not needed when the build fails")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"id": "a", "family": "g", "path": "input.npz",
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    anchor = tmp_path / "calibration.json"
    anchor.write_text(json.dumps({"status": "calibrated", "p_ref": 1.0, "g_ref": 1.0, "time_limits": {"a": 1.0},
                                 "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    reward = tmp_path / "reward.json"
    main(["--submission", str(source), "--manifest", str(manifest), "--calibration", str(anchor),
          "--output", str(tmp_path / "summary.json"), "--reward", str(reward)])
    values = json.loads(reward.read_text())
    assert values["reward"] == 0
    assert all(isinstance(value, (int, float)) for value in values.values())
