"""Small correctness checks for the draft IHVP task, never a timed benchmark.

The examples are synthetic. No private case, imported prediction or historical
wall-clock calibration is needed to exercise the release score boundary.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / "tasks/ihvp-sequence/versions/v1"
GATES = ("amd64_offline", "worker_isolation", "reference_headroom_25pct",
         "two_run_stability", "non_saturation")


@pytest.fixture(scope="module")
def scoring():
    path = VERSION / "grader/scoring.py"
    spec = importlib.util.spec_from_file_location("ihvp_sequence_scoring_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def calibration():
    return {"status": "active", "runtime_fingerprint": "synthetic-runtime-v1",
            "data_fingerprint": "synthetic-data-v1", "E0": .2, "E5": .08,
            "release_gates": {name: True for name in GATES}}


def scored(scoring, result, calibration):
    return scoring.score_result(result, calibration,
                                runtime_fingerprint="synthetic-runtime-v1",
                                data_fingerprint="synthetic-data-v1")


@pytest.mark.parametrize("raw,expected", [
    (0., 100.), (.04, 75.), (.08, 50.), (.14, 25.), (.2, 0.), (1., 0.),
])
def test_native_and_repository_score_breakpoints_and_midpoints(scoring, raw, expected):
    assert math.isclose(scoring.final_score(raw, .2, .08), expected, abs_tol=1e-12)
    assert math.isclose(scoring.native_score(raw, .2, .08), expected / 10., abs_tol=1e-12)


@pytest.mark.parametrize("raw", [-1., float("nan"), float("inf"), -float("inf"), None, True, "0.1"])
def test_scalar_mapping_rejects_invalid_raw_metrics(scoring, raw):
    with pytest.raises((ValueError, TypeError)):
        scoring.native_score(raw, .2, .08)
    with pytest.raises((ValueError, TypeError)):
        scoring.final_score(raw, .2, .08)


@pytest.mark.parametrize("weak,strong", [(.08, .08), (.05, .08), (.2, 0.),
                                         (.2, -.1), (float("inf"), .08),
                                         (.2, float("nan")), (True, .08)])
def test_scalar_mapping_rejects_unordered_or_nonfinite_anchors(scoring, weak, strong):
    with pytest.raises((ValueError, TypeError)):
        scoring.final_score(.01, weak, strong)


def test_valid_zero_vector_loss_is_legal_even_when_score_is_zero(scoring, calibration):
    result = scored(scoring, {"status": "ok", "raw_error": 1.}, calibration)
    assert result["status"] == "ok"
    assert result["raw_error"] == 1.
    assert result["score"] == result["native_score"] == 0.


def test_eligible_reference_and_exact_outputs_map_to_50_and_100(scoring, calibration):
    reference = scored(scoring, {"status": "ok", "raw_error": .08}, calibration)
    exact = scored(scoring, {"status": "ok", "raw_error": 0.}, calibration)
    assert reference["score"] == 50. and reference["native_score"] == 5.
    assert exact["score"] == 100. and exact["native_score"] == 10.


def test_imported_package_score_cannot_override_local_raw_metric(scoring, calibration):
    result = scored(scoring, {"status": "ok", "raw_error": .2,
                              "package_score": 10., "score": 100.}, calibration)
    assert result["score"] == 0.


@pytest.mark.parametrize("status", ["infrastructure_failure", "missing_data"])
def test_platform_failures_are_unmeasured_not_zero(scoring, calibration, status):
    result = scored(scoring, {"status": status, "raw_error": 0., "reason": "synthetic failure"}, calibration)
    assert result["status"] == "unmeasured"
    assert result["score"] is None


def test_candidate_failure_is_zero_only_with_a_qualified_platform(scoring, calibration):
    failure = {"status": "submission_failure", "raw_error": None, "reason": "invalid vector"}
    eligible = scored(scoring, failure, calibration)
    assert eligible["status"] == "submission_failure" and eligible["score"] == 0.
    absent = scored(scoring, failure, None)
    assert absent["status"] == "unmeasured" and absent["score"] is None
    pending = copy.deepcopy(calibration)
    pending["status"] = "measured_pending_activation"
    result = scored(scoring, failure, pending)
    assert result["status"] == "unmeasured" and result["score"] is None


@pytest.mark.parametrize("status", ["ok", "submission_failure"])
def test_missing_calibration_never_falls_back_to_historical_constants(scoring, status):
    result = scored(scoring, {"status": status, "raw_error": 0.}, None)
    assert result["status"] == "unmeasured" and result["score"] is None


@pytest.mark.parametrize("fingerprint", ["runtime_fingerprint", "data_fingerprint"])
def test_runtime_and_dataset_binding_are_required(scoring, calibration, fingerprint):
    calibration[fingerprint] = "another-measurement"
    result = scored(scoring, {"status": "ok", "raw_error": .08}, calibration)
    assert result["status"] == "unmeasured" and result["score"] is None


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("present", [False, True])
def test_every_release_gate_is_required(scoring, calibration, gate, present):
    if present:
        calibration["release_gates"][gate] = False
    else:
        del calibration["release_gates"][gate]
    result = scored(scoring, {"status": "ok", "raw_error": .08}, calibration)
    assert result["status"] == "unmeasured" and result["score"] is None


def test_measurement_pending_activation_cannot_award_formal_scores(scoring, calibration):
    calibration["status"] = "measured_pending_activation"
    result = scored(scoring, {"status": "ok", "raw_error": .08}, calibration)
    assert result["status"] == "unmeasured" and result["score"] is None


@pytest.mark.parametrize("raw", [-1., float("nan"), float("inf"), None])
def test_corrupt_success_metrics_do_not_become_candidate_failures(scoring, calibration, raw):
    result = scored(scoring, {"status": "ok", "raw_error": raw}, calibration)
    assert result["score"] is None and result["status"] == "unmeasured"


@pytest.mark.parametrize("weak,strong", [(.08, .08), (.2, 0.), (float("nan"), .08)])
def test_corrupt_calibration_cannot_activate(scoring, calibration, weak, strong):
    calibration.update(E0=weak, E5=strong)
    result = scored(scoring, {"status": "ok", "raw_error": .01}, calibration)
    assert result["score"] is None and result["status"] == "unmeasured"


@pytest.fixture(scope="module")
def runtime_api():
    path = VERSION / "grader/runtime/api.py"
    spec = importlib.util.spec_from_file_location("ihvp_sequence_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinySequence:
    """Host-only three-stage fixture; no private dataset or generator dependency."""
    dimension, steps, mu, L = 2, 3, 1., 6.

    def __init__(self):
        self.matrices = np.array([[[1., 0.], [0., 4.]], [[3., 1.], [1., 2.]],
                                  [[2., -.25], [-.25, 5.]]])
        self.reference = np.array([[1., 2.], [-1., .5], [.25, -2.]])
        self.right_sides = np.einsum("tij,tj->ti", self.matrices, self.reference)
        self.calls = []

    def rhs(self, step):
        return self.right_sides[step]

    def phase(self, step):
        return int(step == 2)

    def hvp(self, step, vector):
        self.calls.append((step, vector.shape))
        return self.matrices[step] @ vector

    def score(self, step, prediction):
        error = prediction - self.reference[step]
        matrix, b = self.matrices[step], self.right_sides[step]
        return {"energy_loss": float(error @ matrix @ error / (self.reference[step] @ b)),
                "relative_residual": float(np.linalg.norm(matrix @ prediction - b) / np.linalg.norm(b))}


def submission(tmp_path, source):
    directory = tmp_path / "submission"
    directory.mkdir()
    (directory / "solver.py").write_text(textwrap.dedent(source))
    return directory


def test_tiny_online_solve_counts_all_probes_and_resets_each_sequence(runtime_api, tmp_path):
    source = submission(tmp_path, '''
        import numpy as np
        made = 0
        class Solver:
            def __init__(self):
                self.seen = 0
                self.output = np.zeros(2)
            def solve(self, hvp, b, context):
                assert context["step"] == self.seen
                assert set(context) == {"step", "steps", "phase", "phase_changed", "mu", "L",
                                        "deadline", "sequence_deadline", "hard_deadline", "target_rtol"}
                assert context["steps"] == 3
                assert context["phase"] == int(self.seen == 2)
                assert context["phase_changed"] == (self.seen in (0, 2))
                assert b.shape == (2,)
                current = hvp(np.eye(2))
                self.output[:] = np.linalg.solve(current, b)
                self.seen += 1
                return self.output, {"seen": self.seen}
        def make_solver(dimension, seed, options):
            global made
            made += 1
            assert made == 1 and dimension == 2 and seed == 777 and options == {}
            return Solver()
    ''')
    case = TinySequence()
    for _ in range(2):
        result = runtime_api.run_case(source, case, seed=777, wall_seconds=5., memory_mib=256.)
        assert result["valid"] is True and result["completed_steps"] == 3
        assert result["hvp_calls"] == 6
        assert [r["hvp_requests"] for r in result["records"]] == [1, 1, 1]
        assert [r["hvp_vectors"] for r in result["records"]] == [2, 2, 2]
        assert result["mean_energy_loss"] < 1e-28
        # Reusing and mutating the worker's output buffer must not alter old submissions.
        assert np.allclose(result["prediction_array"], case.reference, rtol=0., atol=1e-14)
    assert case.calls == [(t, (2, 2)) for _ in range(2) for t in range(3)]


@pytest.mark.parametrize("action", ["future_hvp", "future_step", "hvp_after_output", "duplicate_output"])
def test_rpc_rejects_future_access_and_changes_after_committing_output(runtime_api, tmp_path, action):
    source = submission(tmp_path, f'''
        import sys
        import numpy as np
        import contracts as protocol
        class Solver:
            def solve(self, hvp, b, context):
                outgoing = sys.__stdout__.buffer
                step = context["step"]
                action = {action!r}
                if action == "future_hvp":
                    protocol.send_message(outgoing, "hvp", np.ones(2), step=step+1)
                elif action == "future_step":
                    protocol.send_message(outgoing, "next", step=step+1)
                else:
                    protocol.send_message(outgoing, "result", np.zeros(2), step=step)
                    kind = "hvp" if action == "hvp_after_output" else "result"
                    protocol.send_message(outgoing, kind, np.zeros(2), step=step)
                # Wait for the host to reject the illegal request, rather than sending another result.
                protocol.receive_message(sys.stdin.buffer)
                return np.zeros(2)
        def make_solver(dimension, seed, options):
            return Solver()
    ''')
    case = TinySequence()
    result = runtime_api.run_case(source, case, wall_seconds=5., memory_mib=256.)
    assert result["valid"] is False
    assert result["termination_reason"].startswith("protocol_failure:")
    assert case.calls == []  # Rejection happens before any unavailable operator is evaluated.


@pytest.fixture(scope="module")
def backend():
    directory = VERSION / "grader/runtime"
    package_name = "ihvp_sequence_backend_testpackage"
    spec = importlib.util.spec_from_file_location(package_name, directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(package_name + ".backend")


def test_dense_energy_definition_zero_exact_and_directional_errors(backend, tmp_path):
    tiny = TinySequence()
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps({"kind": "dense_smoke", "dimension": tiny.dimension,
        "steps": tiny.steps, "mu": tiny.mu, "L": tiny.L, "H": tiny.matrices.tolist(),
        "b": tiny.right_sides.tolist(), "z_star": tiny.reference.tolist(), "phase": [0, 0, 1]}))
    case = backend.Case(path)
    assert case.score(0, np.zeros(2))["energy_loss"] == 1.
    assert case.score(0, tiny.reference[0])["energy_loss"] == 0.
    assert case.score(0, tiny.reference[0] / 2)["energy_loss"] == .25
    assert math.isclose(case.score(0, np.array([1., 0.]))["energy_loss"], 16. / 17.)
    # Teacher failure belongs to the host, not to an innocent solver's accuracy.
    case.data["z_star"][0] = 0.
    with pytest.raises(ArithmeticError):
        case.score(0, np.zeros(2))


def test_signed_dct_operator_and_energy_match_independent_cosine_matrix(backend, tmp_path):
    n, stages = 5, 3
    permutation = np.array([3, 0, 4, 1, 2])
    sign_in = np.array([1., -1., -1., 1., -1.])
    sign_out = np.array([-1., 1., -1., -1., 1.])
    # Construct orthonormal DCT-II analytically, without the runtime FFT helpers.
    cosine = np.sqrt(2. / n) * np.cos(np.pi / n * np.arange(n)[:, None] * (np.arange(n)[None, :] + .5))
    cosine[0] /= np.sqrt(2.)
    q = (sign_out[:, None] * cosine.T * sign_in[None, :]) @ np.eye(n)[permutation]
    spectrum = np.array([1., 2., 4., 8., 16.])
    v = np.zeros((n, 2)); v[:2, 0] = 1. / np.sqrt(2.); v[3:, 1] = 1. / np.sqrt(2.)
    a = np.array([[1., 1.1, 1.2, .9, 1.], [1.1, 1., .9, 1.2, 1.], [.8, 1., 1.1, 1., 1.2]])
    weights = np.array([[.1, .2], [.15, .1], [.05, .25]])
    root = np.diag(np.sqrt(spectrum))
    matrices = [q @ root @ (np.diag(a[t]) + (v * weights[t]) @ v.T) @ root @ q.T for t in range(stages)]
    right_sides = np.array([[1., 2., -1., .5, .75], [-2., .5, .25, 1., -.5], [.25, -1., 2., -.5, 1.]])
    reference = np.array([np.linalg.solve(h, b) for h, b in zip(matrices, right_sides)])
    path = tmp_path / "synthetic_teacher.npz"
    np.savez(path, spectrum=spectrum, V=v, permutation=permutation, sign_in=sign_in, sign_out=sign_out,
             diagonal_weights=a, lowrank_weights=weights, b=right_sides, z_star=reference, phase=[0, 0, 1],
             metadata_json=json.dumps({"schema": "coupled_quadratic_bilevel_ihvp_signed_dct_v1",
                                       "dimension": n, "steps": stages, "mu": .8, "L": 40.}))
    case = backend.Case(path)
    block = np.column_stack((np.arange(n) - 1., np.linspace(-1., 1., n)))
    for t, matrix in enumerate(matrices):
        assert np.allclose(case.hvp(t, block), matrix @ block, rtol=2e-14, atol=2e-14)
        assert np.allclose(case.hvp(t, block[:, 0]), matrix @ block[:, 0], rtol=2e-14, atol=2e-14)
        assert math.isclose(case.score(t, np.zeros(n))["energy_loss"], 1., rel_tol=1e-13)
        assert math.isclose(case.score(t, reference[t] / 2)["energy_loss"], .25, rel_tol=1e-13)


@pytest.fixture(scope="module")
def local_grader():
    path = VERSION / "grader/evaluate.py"
    spec = importlib.util.spec_from_file_location("ihvp_sequence_local_grader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reason", ["unexpected_worker_exit", "nonzero_worker_exit", "wall_limit", "memory_limit",
                                   "protocol_failure: output for unavailable system"])
def test_worker_failures_are_candidate_failures_not_platform_excuses(local_grader, reason):
    value = local_grader.normalize_case_result({"valid": False, "termination_reason": reason, "mean_energy_loss": None})
    assert value["status"] == "submission_failure"


def test_trusted_platform_failure_retains_infrastructure_status(local_grader):
    value = local_grader.normalize_case_result({"valid": False, "infrastructure_failure": True,
                                               "termination_reason": "slot lost", "mean_energy_loss": None})
    assert value["status"] == "infrastructure_failure"


def test_actual_worker_crash_cannot_request_an_unmeasured_candidate_result(runtime_api, local_grader, tmp_path):
    source = submission(tmp_path, '''
        import os
        class Solver:
            def solve(self, hvp, b, context):
                os._exit(17)
        def make_solver(dimension, seed, options):
            return Solver()
    ''')
    result = runtime_api.run_case(source, TinySequence(), wall_seconds=5., memory_mib=256.)
    assert result["valid"] is False and result["termination_reason"] == "unexpected_worker_exit"
    assert local_grader.normalize_case_result(result)["status"] == "submission_failure"


@pytest.mark.parametrize("scope", ["full", "partial", "smoke"])
def test_missing_candidate_entry_preserves_full_vs_diagnostic_scoring(local_grader, calibration, monkeypatch, tmp_path, scope):
    monkeypatch.setattr(local_grader, "identity", lambda: ({"synthetic": True}, "synthetic-runtime-v1"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [{"id": "one", "path": "one.npz"}, {"id": "two", "path": "two.npz"}]}))
    if scope == "smoke":
        calibration["data_fingerprint"] = local_grader.file_sha(VERSION / "data/samples/smoke.json")
    else:
        calibration["data_fingerprint"] = local_grader.file_sha(manifest)
    anchor_path = tmp_path / "calibration.json"
    anchor_path.write_text(json.dumps(calibration))
    empty_source = tmp_path / "empty_submission"
    empty_source.mkdir()
    args = SimpleNamespace(out=tmp_path / "result", smoke=scope == "smoke", manifest=manifest,
                           case_limit=1 if scope == "partial" else None, recalibrate=False,
                           submission=empty_source, calibration=anchor_path, repeats=2,
                           wall_seconds=3. if scope == "smoke" else 30., resume=False)
    payload = local_grader.run(args)
    assert payload["result"]["status"] == "submission_failure"
    assert payload["score"]["score"] == (0. if scope == "full" else None)


@pytest.fixture(scope="module")
def sealer():
    path = ROOT / "scripts/seal/prepare_ihvp_sequence.py"
    spec = importlib.util.spec_from_file_location("ihvp_sequence_sealer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_staging_refuses_a_nonignored_repository_destination(sealer):
    forbidden = VERSION / "data/private/should-not-be-created-by-test"
    assert not forbidden.exists()
    with pytest.raises(ValueError, match="git-ignored"):
        sealer.local_destination(forbidden)
    assert not forbidden.exists()
    allowed = ROOT / "var/raw/ihvp-sequence-v1/test-path-only"
    assert sealer.local_destination(allowed) == allowed.resolve()


def test_private_staging_manifest_strips_generation_seeds_and_source_paths(sealer, tmp_path):
    source = tmp_path / "received_private"
    source.mkdir()
    entries = []
    for i in range(15):
        body = source / f"case_{i:02d}.npz"
        body.write_bytes(f"synthetic transport-only body {i}".encode())
        entries.append({"id": f"case_{i:02d}", "path": body.name, "sha256": sealer.digest_file(body),
                        "dimension": 16384, "steps": 50, "mu": 1., "L": 100.,
                        "family": "synthetic", "profile": "test", "seed": 90000 + i,
                        "seqid": f"synthetic-private-seed-{90000+i}"})
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"cases": entries}))
    destination = tmp_path / "maintainer_only/private"
    staged = sealer.materialize_split(manifest, destination, "private_final", "copy")
    assert staged["split"] == "private_final" and staged["sequence_count"] == 15
    assert len(list(destination.rglob("teacher.npz"))) == 15
    assert all("seed" not in case and "seqid" not in case for case in staged["cases"])
    assert str(source) not in json.dumps(staged)
    assert all(not Path(case["path"]).is_absolute() for case in staged["cases"])


def test_draft_dispatch_never_activates_reference_or_awards_formal_scores(tmp_path):
    from arena.tasks import TASKS
    task = TASKS["ihvp-sequence"]
    assert task.status == "draft"
    parts = {"candidate": {"status": "ok", "raw_error": 0.},
             "weak": {"status": "ok", "raw_error": .2},
             "strong": {"status": "ok", "raw_error": .08}}
    ok, reason = task.baseline_ok({"parts": parts})
    assert ok is False and reason
    result = task.score({}, {"parts": parts}, {"metrics": {"parts": parts}})
    assert result["status"] == "unmeasured" and result["score"] is None
    grade = task.run_grader(tmp_path / "unused_source", tmp_path / "unused_output", None)
    assert grade["status"] == "infrastructure_failure" and grade["raw_error"] is None
