"""Tiny v4 generator -> backend dispatch -> real worker integration checks.

No real cohort or private seed is read. These three local child processes are
protocol correctness checks, not a timing benchmark or isolation certification.
Do not run them alongside reference timing experiments.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
VERSION = ROOT / "tasks/ihvp-sequence/versions/v4"


def load_file(name, path, *, package=False):
    kwargs = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v4_modules():
    # Every runtime module has a unique package name: existing v1/v2 tests may
    # have imported their own unrelated modules named runtime/api/backend.
    runtime_name = "_frontieram_ihvp_v4_integration_runtime"
    runtime_dir = VERSION / "grader/runtime"
    package = load_file(runtime_name, runtime_dir / "__init__.py", package=True)
    api = importlib.import_module(runtime_name + ".api")
    backend = importlib.import_module(runtime_name + ".backend")
    scoring = load_file("_frontieram_ihvp_v4_integration_scoring", VERSION / "grader/scoring.py")
    model = load_file("_frontieram_ihvp_v4_integration_model", VERSION / "curator/model.py")
    original_path = list(sys.path)
    try:
        # The production CLI uses absolute imports. Bind those names only while
        # importing it, then restore them so tests do not contaminate one another.
        with patch.dict(sys.modules, {"runtime": package, "runtime.api": api,
                                      "runtime.backend": backend, "scoring": scoring,
                                      "model": model}):
            generator = load_file("_frontieram_ihvp_v4_integration_generator", VERSION / "curator/generate.py")
            evaluator = load_file("_frontieram_ihvp_v4_integration_evaluator", VERSION / "grader/evaluate.py")
    finally:
        sys.path[:] = original_path
    return SimpleNamespace(api=api, backend=backend, generator=generator, evaluator=evaluator)


@pytest.fixture(scope="module")
def tiny_teacher(v4_modules, tmp_path_factory):
    # This fixed PUBLIC UNIT-TEST seed is unrelated to either frozen data split.
    configuration = {"id": "synthetic_integration", "split": "selftest",
                     "seed": 314159265, "profile": "unit_test", "condition_target": 1e5,
                     "settings": {**v4_modules.generator.FAMILIES[0], "architecture":"routed_only"}}
    arrays, _ = v4_modules.generator.generate_arrays(configuration, dimension=64, steps=4)
    path = tmp_path_factory.mktemp("ihvp_v4_synthetic_64x4") / "teacher.npz"
    np.savez(path, **arrays)
    return path


@pytest.fixture(autouse=True)
def local_protocol_only(monkeypatch):
    # UID/mount isolation is tested on the actual grading platform, separately.
    monkeypatch.delenv("IHVP_WORKER_UID", raising=False)


def submission(tmp_path, source):
    directory = tmp_path / "submission"
    directory.mkdir()
    (directory / "solver.py").write_text(textwrap.dedent(source))
    return directory


def load_tiny_case(v4_modules, tiny_teacher):
    case = v4_modules.backend.Case(tiny_teacher)
    assert case.meta["schema"] == "frontieram/ihvp-routed-factored-spd-4"
    assert case.dimension == 64 and case.steps == 4
    assert callable(case.rhs) and callable(case.reference) and callable(case.phase)
    return case


def test_v4_dispatch_and_worker_accept_legal_zero_vectors(v4_modules, tiny_teacher, tmp_path):
    source = submission(tmp_path, '''
        import numpy as np
        class Solver:
            def solve(self, hvp, b, context):
                return np.zeros_like(b)
        def make_solver(dimension, seed, options):
            assert dimension == 64
            return Solver()
    ''')
    case = load_tiny_case(v4_modules, tiny_teacher)
    result = v4_modules.api.run_case(source, case, wall_seconds=5., memory_mib=256.)
    assert result["valid"] is True and result["completed_steps"] == 4
    assert result["mean_energy_loss"] == 1.
    assert result["hvp_calls"] == 0
    assert v4_modules.evaluator.normalize_case_result(result)["status"] == "ok"


def test_v4_worker_can_solve_tiny_systems_through_legal_block_probes(v4_modules, tiny_teacher, tmp_path):
    source = submission(tmp_path, '''
        import numpy as np
        class Solver:
            def __init__(self):
                self.next_step = 0
            def solve(self, hvp, b, context):
                assert context["step"] == self.next_step
                self.next_step += 1
                matrix = hvp(np.eye(len(b)))
                return np.linalg.solve(matrix, b), {"strategy": "tiny exact HVP-basis solve"}
        def make_solver(dimension, seed, options):
            assert dimension == 64
            return Solver()
    ''')
    case = load_tiny_case(v4_modules, tiny_teacher)
    result = v4_modules.api.run_case(source, case, wall_seconds=5., memory_mib=256.)
    assert result["valid"] is True and result["completed_steps"] == 4
    assert result["mean_energy_loss"] < 1e-16
    assert result["hvp_calls"] == 64 * 4
    assert [r["hvp_requests"] for r in result["records"]] == [1] * 4
    assert [r["hvp_vectors"] for r in result["records"]] == [64] * 4
    assert result["fraction_residual_target_met"] == 1.


def test_v4_huge_finite_predictions_are_submission_failures(v4_modules, tiny_teacher, tmp_path):
    source = submission(tmp_path, '''
        import numpy as np
        class Solver:
            def solve(self, hvp, b, context):
                return np.full_like(b, 1e308)
        def make_solver(dimension, seed, options):
            return Solver()
    ''')
    case = load_tiny_case(v4_modules, tiny_teacher)
    result = v4_modules.api.run_case(source, case, wall_seconds=5., memory_mib=256.)
    assert result["completed_steps"] == 4
    assert result["valid"] is False and result["termination_reason"] == "invalid_prediction"
    assert result["mean_energy_loss"] is None
    assert all(row["energy_loss"] is None for row in result["records"])
    normalized = v4_modules.evaluator.normalize_case_result(result)
    assert normalized["status"] == "submission_failure"
    assert normalized["status"] != "infrastructure_failure"
