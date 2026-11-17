import json
from pathlib import Path

import numpy as np
import pytest

from qpbench.check import check_solution
from qpbench.evaluate import checked_output, read_manifest
from qpbench.model import Problem, SparseMatrix, save_output
from qpbench.scoring import aggregate_cases, compute_reward
from grade import calibration

VERSION = Path(__file__).resolve().parents[2]


def problem():
    matrix = SparseMatrix(np.array([1.0]), np.array([0]), np.array([0, 1]), (1, 1))
    return Problem(matrix, matrix, np.array([-2.0]), np.array([1.0]), 0.0, {})


def rows(outcomes=("accepted",) * 3, seconds=1.0):
    return [{"case_id": "a", "group": "g", "repeats": [
        {"repeat": i + 1, "outcome": outcome, "status": "solved" if outcome == "accepted" else None,
         "elapsed_seconds": seconds} for i, outcome in enumerate(outcomes)]}]


def test_independent_kkt_certificate_and_slack_complementarity():
    qp = problem()
    assert check_solution(qp, {"metadata": {"status": "solved"}, "x": [1.0], "y": [1.0]})["passed"]
    # Feasible and stationary, but the positive multiplier on a slack row is invalid.
    report = check_solution(qp, {"metadata": {"status": "solved"}, "x": [0.0], "y": [2.0]})
    assert report["stationarity"] == 0
    assert report["complementarity"] > 0
    assert not report["passed"]


def test_output_snapshot_rejects_links_and_nonfinite(tmp_path):
    path = tmp_path / "result.npz"
    save_output(path, np.array([1.0]), np.array([1.0]))
    assert checked_output(path, problem())["passed"]
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(OSError):
        checked_output(link, problem())
    save_output(path, np.array([np.inf]), np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        checked_output(path, problem())


def test_mapping_and_failed_repeat_penalty():
    summary = aggregate_cases(rows(), expected_case_ids=["a"], time_limits={"a": 10.0})
    anchor = {"p_ref": 1.0, "g_ref": 1.0}
    assert compute_reward(summary, anchor)["score"] == 50
    assert compute_reward({**summary, "g": 0.001}, anchor)["score"] > 99
    assert compute_reward({**summary, "g": 2.0}, anchor)["score"] < 50
    failure = aggregate_cases(rows(("accepted", "algorithm_failure", "accepted")),
                              expected_case_ids=["a"], time_limits={"a": 10.0})
    assert failure["valid"] and failure["p"] == 0
    assert failure["g"] == pytest.approx(10.0)
    assert compute_reward(failure, anchor)["score"] == 0
    with pytest.raises(ValueError):
        compute_reward(summary, {"p_ref": 0.5, "g_ref": 1.0})


@pytest.mark.parametrize("broken", [[], rows() * 2, rows(("evaluation_error",) * 3)])
def test_missing_duplicate_and_infrastructure_results_cannot_be_scored(broken):
    summary = aggregate_cases(broken, expected_case_ids=["a"], time_limits={"a": 10.0})
    assert not summary["valid"]
    with pytest.raises(ValueError):
        compute_reward(summary, {"p_ref": 1.0, "g_ref": 1.0})


def test_calibration_uses_same_median_and_requires_headroom(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    raw = rows()
    for repeat, seconds in zip(raw[0]["repeats"], (1.0, 2.0, 3.0)):
        repeat["elapsed_seconds"] = seconds
    summary = aggregate_cases(raw, expected_case_ids=["a"], time_limits={"a": 60.0})
    summary["raw_cases"] = raw
    anchor = calibration(summary, manifest)
    assert anchor["g_ref"] == pytest.approx(2.0)
    assert anchor["time_limits"]["a"] == 4.5
    raw[0]["repeats"][0]["elapsed_seconds"] = 55.0
    with pytest.raises(ValueError, match="headroom"):
        calibration(summary, manifest)


def test_public_assets_are_complete_and_samples_are_subset():
    public = read_manifest(VERSION / "data/public/manifest.json")
    samples = read_manifest(VERSION / "data/samples/manifest.json")
    assert len(public["cases"]) == 64
    assert len(samples["cases"]) == 4
    assert len({r["family"] for r in samples["cases"]}) == 4
    assert {r["sha256"] for r in samples["cases"]} <= {r["sha256"] for r in public["cases"]}


def test_tampered_trusted_input_is_infrastructure_error(tmp_path):
    (tmp_path / "input.npz").write_bytes(b"changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"id": "a", "path": "input.npz", "sha256": "bad"}]}))
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(manifest)
