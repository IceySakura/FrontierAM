"""Independent tiny linear-algebra checks for hard-sequence diagnostics.

These check mathematical identities, causality, and metric interpretation.
They do not encode thresholds asserting that a generated benchmark is hard.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def diagnostics():
    path = ROOT / "scripts/analyze/ihvp_hard_diagnostics.py"
    spec = importlib.util.spec_from_file_location("ihvp_hard_diagnostics_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DenseCase:
    def __init__(self, matrices, references):
        self.matrices = np.asarray(matrices, dtype=float)
        self.references = np.asarray(references, dtype=float)
        self.steps, self.dimension = self.references.shape
        self.b = np.einsum("tij,tj->ti", self.matrices, self.references)
        self.mu = min(np.linalg.eigvalsh(h)[0] for h in self.matrices) * .99
        self.L = max(np.linalg.eigvalsh(h)[-1] for h in self.matrices) * 1.01
        self.meta = {"schema": "synthetic-dense", "family": "unit-test"}

    def reference(self, step):
        return self.references[step]

    def rhs(self, step):
        return self.b[step]

    def hvp(self, step, vector):
        return self.matrices[step] @ vector


def test_galerkin_uses_current_energy_metric_not_euclidean_projection(diagnostics):
    h = np.array([[4., 1.], [1., 2.]])
    target = np.array([0., 1.])
    history = np.array([[1.], [0.]])
    result = diagnostics.galerkin_projection(target, history, lambda v: h @ v)
    # Galerkin coefficient is 1/4. Euclidean projection gives zero and loss one.
    assert result["basis_rank"] == 1
    assert np.isclose(result["relative_H_energy_error_squared"], 7. / 8.)
    assert result["coarse_orthogonality_relative"] < 1e-14


def test_dependent_history_has_no_invented_completion_vectors(diagnostics):
    h = np.diag([1., 4., 9.])
    history = np.array([[1., 1e6, -2.], [0., 0., 0.], [0., 0., 0.]])
    result = diagnostics.galerkin_projection(np.array([0., 1., 0.]), history, lambda v: h @ v)
    assert result["columns"] == 3 and result["basis_rank"] == 1
    assert result["relative_H_energy_error_squared"] == 1.


def test_exact_past_span_and_empty_history_endpoints(diagnostics):
    h = np.array([[3., .5], [.5, 2.]])
    target = np.array([1., -2.])
    empty = diagnostics.galerkin_projection(target, np.empty((2, 0)), lambda v: h @ v)
    exact = diagnostics.galerkin_projection(target, np.column_stack((target, target * 2)), lambda v: h @ v)
    assert empty["relative_H_energy_error_squared"] == 1.
    assert exact["relative_H_energy_error_squared"] < 1e-28


def test_temporal_participation_and_tails_have_known_endpoints(diagnostics):
    orthogonal = diagnostics.normalized_gram_spectrum(np.eye(50))
    repeated = diagnostics.normalized_gram_spectrum(np.ones((50, 50)))
    assert np.isclose(orthogonal["participation_ratio"], 50.)
    assert np.isclose(orthogonal["entropy_effective_rank"], 50.)
    assert np.isclose(orthogonal["tail_energy_fraction"]["16"], 34. / 50.)
    assert np.isclose(orthogonal["tail_energy_fraction"]["48"], 2. / 50.)
    assert np.isclose(repeated["participation_ratio"], 1.)
    assert repeated["numerical_rank"] == 1
    assert repeated["tail_energy_fraction"]["16"] < 1e-14


def test_fixed_metric_rank_is_invariant_to_a_coordinate_change(diagnostics):
    x = np.array([[1., .5, -1.], [2., 0., .5]])
    metric = np.array([[4., 1.], [1., 2.]])
    change = np.array([[2., .25], [-.5, 1.]])
    inverse = np.linalg.inv(change)
    transformed_metric = inverse.T @ metric @ inverse
    first = diagnostics.normalized_gram_spectrum(x.T @ metric @ x)
    second = diagnostics.normalized_gram_spectrum((change @ x).T @ transformed_metric @ (change @ x))
    assert np.isclose(first["participation_ratio"], second["participation_ratio"])
    assert np.allclose(first["normalized_singular_values"], second["normalized_singular_values"], atol=3e-8)


def test_fifty_steps_give_only_two_full_history48_windows(diagnostics):
    references = np.eye(50)
    case = DenseCase(np.repeat(np.eye(50)[None, :, :], 50, axis=0), references)
    result = diagnostics.diagnose_case(case, probe_count=2)
    assert result["history_projection"]["16"]["full_windows_only"]["count"] == 34
    assert result["history_projection"]["48"]["full_windows_only"]["count"] == 2
    assert result["history_projection"]["48"]["full_window_steps_zero_based"] == [48, 49]
    assert result["normalized_temporal_rank_fixed_H0"]["maximum_possible_rank"] == 50
    assert all(row["history"]["48"]["relative_H_energy_error_squared"] == 1. for row in result["per_step"])


def test_future_references_do_not_enter_earlier_history_projections(diagnostics):
    rng = np.random.default_rng(47)
    n, stages = 7, 6
    matrices = []
    for _ in range(stages):
        column = rng.standard_normal(n)
        matrices.append(np.diag(np.arange(1., n + 1)) + np.outer(column, column) / n)
    references = rng.standard_normal((stages, n))
    original = diagnostics.diagnose_case(DenseCase(matrices, references), histories=(2, 4), probe_count=2)
    changed = references.copy(); changed[-1] *= -100.
    modified = diagnostics.diagnose_case(DenseCase(matrices, changed), histories=(2, 4), probe_count=2)
    for left, right in zip(original["per_step"][:-1], modified["per_step"][:-1]):
        assert left["history"] == right["history"]
        assert left["warm_start_relative_H_energy_error_squared"] == right["warm_start_relative_H_energy_error_squared"]
    assert original["history_projection"]["2"]["full_windows_only"]["count"] == 4
    assert original["history_projection"]["4"]["full_windows_only"]["count"] == 2


def test_noncommuting_probe_and_commuting_control(diagnostics):
    references = np.array([[1., 1.], [1., -.5], [.5, 1.]])
    commuting = DenseCase([np.diag([1., 2.]), np.diag([2., 3.]), np.diag([3., 2.])], references)
    noncommuting = DenseCase([np.diag([1., 2.]), np.array([[2., .5], [.5, 3.]]), np.diag([3., 2.])], references)
    first = diagnostics.diagnose_case(commuting, histories=(1,), probe_count=2)
    second = diagnostics.diagnose_case(noncommuting, histories=(1,), probe_count=2)
    assert first["adjacent_summary"]["sampled_relative_commutator"]["max"] == 0.
    assert second["adjacent_summary"]["sampled_relative_commutator"]["min"] > 0.


def test_bad_teacher_is_rejected_instead_of_becoming_a_hardness_result(diagnostics):
    case = DenseCase([np.eye(2), np.eye(2)], np.array([[1., 1.], [1., -1.]]))
    case.b[1] += 1.
    with pytest.raises(ValueError, match="reference residual"):
        diagnostics.diagnose_case(case, histories=(1,), probe_count=2)


def test_zero_rhs_is_outside_relative_loss_contract(diagnostics):
    case = DenseCase([np.eye(2), np.eye(2)], np.array([[1., 1.], [1., -1.]]))
    case.b[1] = 0.
    with pytest.raises(ValueError, match="zero RHS"):
        diagnostics.diagnose_case(case, histories=(1,), probe_count=2)


def _rotation_matrix(n, permutation, angles):
    result = np.zeros((n, n))
    for i, theta in enumerate(angles):
        first, second = permutation[2 * i:2 * i + 2]
        c, s = np.cos(theta), np.sin(theta)
        result[first, first], result[first, second] = c, -s
        result[second, first], result[second, second] = s, c
    return result


def _factored_fixture(architecture):
    """Construct tiny references via dense F solves, independently of model.py."""
    rng, n, steps = np.random.default_rng(314159), 8, 4
    depth = 1 if architecture == "butterfly" else (2 if architecture.endswith("2") else 3)
    permutations = np.stack([rng.permutation(n) for _ in range(depth)])
    signs = rng.choice([-1., 1.], size=(depth, n))
    angles = rng.uniform(-2., 2., size=(3, n // 2)) if depth == 1 else np.empty((0, n // 2))
    temporal_p = np.stack([rng.permutation(n), rng.permutation(n)])
    temporal = rng.uniform(-.2, .2, size=(steps, 2, n // 2))
    block_p = rng.permutation(n)
    theta = rng.uniform(-1., 1., size=(steps, n // 2))
    c, s = np.cos(theta), np.sin(theta)
    coefficient = np.stack((.85 * c * c + 1.15 * s * s, (.85 - 1.15) * c * s,
                             .85 * s * s + 1.15 * c * c), axis=-1)
    diagonal = np.geomspace(1., 1000., n)[None, :] * rng.uniform(.95, 1.05, size=(steps, n))
    if architecture == "butterfly":
        base = np.eye(n)[permutations[0]]
        for j in range(3):
            positions = np.arange(n).reshape(-1, 2, 1 << j)
            pairing = np.column_stack((positions[:, 0].ravel(), positions[:, 1].ravel())).ravel()
            base = _rotation_matrix(n, pairing, angles[j]) @ base
        base = np.diag(signs[0]) @ base
    else:
        cosine = np.sqrt(2. / n) * np.cos(np.pi / n * np.arange(n)[:, None] * (np.arange(n)[None, :] + .5))
        cosine[0] /= np.sqrt(2.)
        base = np.eye(n)
        for p, sign in zip(permutations, signs):
            base = cosine @ np.diag(sign) @ np.eye(n)[p] @ base
    factors, q_matrices = [], []
    for t in range(steps):
        q = base @ _rotation_matrix(n, temporal_p[1], temporal[t, 1]) @ _rotation_matrix(n, temporal_p[0], temporal[t, 0])
        b = np.zeros((n, n))
        for j, (a, off, d) in enumerate(coefficient[t]):
            first, second = block_p[2*j:2*j+2]
            b[first, first], b[first, second] = a, off
            b[second, first], b[second, second] = off, d
        factors.append(b @ np.diag(np.sqrt(diagonal[t])) @ q.T)
        q_matrices.append(q)
    y = rng.standard_normal((steps, n))
    reference = np.array([np.linalg.solve(f, target) for f, target in zip(factors, y)])
    rhs = np.array([f.T @ target for f, target in zip(factors, y)])
    arrays = {"base_angles": angles, "base_permutations": permutations, "base_signs": signs,
              "temporal_permutations": temporal_p, "temporal_angles": temporal,
              "block_permutation": block_p, "block_coefficients": coefficient,
              "diagonal": diagonal, "whitened_rhs": y, "b": rhs, "z_star": reference,
              "phase": np.array([0, 0, 1, 1]), "metadata_json": np.array(json.dumps({
                  "schema": "frontieram/ihvp-hard-factored-spd-1", "dimension": n, "steps": steps,
                  "mu": .6, "L": 1600., "architecture": architecture}))}
    return arrays, factors, q_matrices


@pytest.mark.parametrize("architecture", ["butterfly", "compound_dct2", "compound_dct3"])
def test_independent_hard_factors_match_dense_definitions(diagnostics, architecture):
    arrays, factors, qs = _factored_fixture(architecture)
    case = diagnostics.IndependentHardCase(arrays)
    block = np.arange(16.).reshape(8, 2) / 16.
    for t, (factor, q) in enumerate(zip(factors, qs)):
        assert np.allclose(case.q_rows(t, np.eye(8)).T, q, rtol=1e-13, atol=1e-13)
        assert np.allclose(case.q_rows(t, np.eye(8), True).T, q.T, rtol=1e-13, atol=1e-13)
        assert np.allclose(case.factor_rows(t, np.eye(8)).T, factor, rtol=1e-13, atol=1e-13)
        assert np.allclose(case.hvp(t, block), factor.T @ factor @ block, rtol=1e-12, atol=1e-12)
        assert np.allclose(case.independent_reference(t), case.reference(t), rtol=1e-12, atol=1e-12)
        assert case.score(t, case.reference(t))["energy_loss"] == 0.
        assert case.score(t, np.zeros(8))["energy_loss"] == 1.
        assert np.isclose(case.score(t, .5 * case.reference(t))["energy_loss"], .25)
        assert case.reference_factor_loss(t) < 1e-27


@pytest.mark.parametrize("architecture", ["butterfly", "compound_dct2", "compound_dct3"])
def test_production_hard_model_matches_independent_dense_fixture(diagnostics, architecture):
    path = ROOT / "tasks/ihvp-sequence/versions/v3/curator/model.py"
    spec = importlib.util.spec_from_file_location("ihvp_hard_production_model_under_test", path)
    model = importlib.util.module_from_spec(spec); spec.loader.exec_module(model)
    arrays, factors, _ = _factored_fixture(architecture)
    case = model.HardCase(arrays)
    probes = np.column_stack((np.arange(8.) / 8., np.linspace(-1., 1., 8)))
    for t, factor in enumerate(factors):
        assert np.allclose(case.hvp(t, probes), factor.T @ factor @ probes, rtol=1e-12, atol=1e-12)
        assert np.allclose(case.reference(t), np.linalg.solve(factor, arrays["whitened_rhs"][t]), rtol=1e-12, atol=1e-12)
        assert np.isclose(case.score(t, .5 * case.reference(t))["energy_loss"], .25)
        assert case.score(t, case.reference(t))["energy_loss"] == 0.
    # This also verifies the callable rhs/reference contract expected by run_case.
    diagnosis = diagnostics.diagnose_case(case, histories=(2,), probe_count=2)
    assert diagnosis["steps"] == 4 and diagnosis["max_reference_relative_residual"] < 1e-12
