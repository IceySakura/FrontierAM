#!/usr/bin/env python3
"""Private, post-generation diagnostics for related SPD linear systems.

The linear algebra below uses only rhs(t), reference(t), hvp(t,v), dimension,
steps and optional mu/L metadata. It imports no solver or grader backend.
History projection is an oracle diagnostic using TRUE PAST solutions, not a
feasible contestant, and it is not a lower bound on all recycling methods.
Run outside timed solver experiments. Only the requested report is written.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.fft import dct, idct


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value, label):
    array = np.asarray(value, dtype=float)
    require(np.isfinite(array).all(), f"nonfinite {label}")
    return array


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator == 0 else float(np.clip(left @ right / denominator, -1., 1.))


def summarize(values):
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if not array.size:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
    return {"count": int(array.size), "mean": float(array.mean()), "min": float(array.min()),
            "median": float(np.median(array)), "max": float(array.max())}


def series_row(case, name, step):
    value = getattr(case, name)
    return value(step) if callable(value) else value[step]


class IndependentHardCase:
    """Independent ROW-oriented factor implementation, without model helpers.

    For the hard-v4 schema F=B sqrt(D) Q.T, H=F.T F. This implementation
    independently validates permutations/signs/SPD blocks and uses the stable
    positive-factor energy. Tiny tests reconstruct Q analytically as a dense
    matrix; large audits never materialize an n-by-n matrix.
    """
    def __init__(self, path):
        if isinstance(path, (str, Path)):
            with np.load(path, allow_pickle=False) as archive:
                self.arrays = {key: archive[key] for key in archive.files}
        else:
            self.arrays = dict(path)
        a = self.arrays
        self.meta = json.loads(str(np.asarray(a["metadata_json"]).item()))
        require(self.meta["schema"] == "frontieram/ihvp-routed-factored-spd-4", "unexpected hard teacher schema")
        self.dimension, self.steps = int(self.meta["dimension"]), int(self.meta["steps"])
        self.mu, self.L = float(self.meta["mu"]), float(self.meta["L"])
        n, t = self.dimension, self.steps
        require(n > 0 and n % 2 == 0 and t > 1, "invalid hard dimensions")
        for key, shape in {"diagonal": (t, n), "b": (t, n), "z_star": (t, n),
                           "whitened_rhs": (t, n), "phase": (t,),
                           "temporal_angles": (t, 2, n // 2),
                           "temporal_permutations": (2, n),
                           "block_permutation": (n,), "block_coefficients": (t, n // 2, 3)}.items():
            require(a[key].shape == shape, "teacher shape mismatch: " + key)
        for key, value in a.items():
            if value.dtype.kind in "fc":
                finite(value, key)
        require(np.all(a["diagonal"] > 0), "nonpositive diagonal")
        for key in ("temporal_permutations", "block_permutation", "base_permutations", "base_routing"):
            values = np.atleast_2d(a[key])
            require(values.dtype.kind in "iu", "noninteger permutation")
            require(all(np.array_equal(np.sort(p), np.arange(n)) for p in values), "invalid " + key)
        require(a["base_signs"].shape == a["base_permutations"].shape, "base sign shape differs")
        require(np.isin(a["base_signs"], [-1., 1.]).all(), "nonorthogonal base signs")
        require(a["base_angles"].shape == (len(a["base_routing"]), n//2), "invalid routed layers")
        require(a["base_routing"].ndim == 2 and 1 <= len(a["base_routing"]) <= 32, "invalid routing depth")
        require(a["post_signs"].shape == (n,) and np.isin(a["post_signs"],[-1,1]).all(), "invalid post signs")
        stages = {"routed_only":0,"routed_dct2":2,"routed_dct3":3}[self.meta["architecture"]]
        require(a["base_permutations"].shape == (stages,n), "invalid DCT stage count")
        aa, bb, cc = np.moveaxis(a["block_coefficients"], -1, 0)
        radius = np.sqrt((aa - cc) ** 2 + 4 * bb ** 2)
        beta_min, beta_max = float(np.min((aa + cc - radius) / 2)), float(np.max((aa + cc + radius) / 2))
        require(beta_min > 0, "B block is not SPD")
        lower = float(a["diagonal"].min()) * beta_min ** 2
        upper = float(a["diagonal"].max()) * beta_max ** 2
        require(0 < self.mu <= lower * (1 + 1e-10), "claimed uniform lower bound is not justified by factors")
        require(self.L >= upper * (1 - 1e-10), "claimed uniform upper bound is not justified by factors")
        require(np.all(np.linalg.norm(a["whitened_rhs"], axis=1) > 0), "zero whitened target")
        self.factor_spectral_certificate = {"B_eigenvalue_min": beta_min, "B_eigenvalue_max": beta_max,
                                            "factor_lower_bound": lower, "factor_upper_bound": upper}

    def rhs(self, t):
        return self.arrays["b"][t]

    def reference(self, t):
        return self.arrays["z_star"][t]

    def phase(self, t):
        return int(self.arrays["phase"][t])

    def _pairs(self, rows, permutation, angles, transpose):
        paired = rows[:, permutation]
        first, second = paired[:, ::2], paired[:, 1::2]
        c, s = np.cos(angles)[None, :], np.sin(angles)[None, :]
        if transpose:
            s = -s
        result = np.empty_like(rows)
        result[:, permutation[::2]] = c * first - s * second
        result[:, permutation[1::2]] = s * first + c * second
        return result

    def _base(self, rows, transpose):
        a = self.arrays
        p, signs = a["base_permutations"], a["base_signs"]
        work = rows * a["post_signs"][None,:] if transpose else rows.copy()
        if not transpose:
            for j, routing in enumerate(a["base_routing"]):
                work = self._pairs(work, routing, a["base_angles"][j], False)
        for j in (range(len(p)-1,-1,-1) if transpose else range(len(p))):
            if transpose:
                work = idct(work,type=2,norm="ortho",axis=1,workers=1) * signs[j][None,:]
                result=np.empty_like(work); result[:,p[j]]=work; work=result
            else:
                work=dct(work[:,p[j]]*signs[j][None,:],type=2,norm="ortho",axis=1,workers=1)
        if transpose:
            for j in range(len(a["base_routing"])-1,-1,-1):
                work=self._pairs(work,a["base_routing"][j],a["base_angles"][j],True)
            return work
        return work * a["post_signs"][None,:]

    def q_rows(self, t, rows, transpose=False):
        # Column Q_t = Q_base R_1(t) R_0(t); each operation acts on row batches.
        work = np.asarray(rows, dtype=float)
        if transpose:
            work = self._base(work, True)
        for j in ((1, 0) if transpose else (0, 1)):
            work = self._pairs(work, self.arrays["temporal_permutations"][j],
                               self.arrays["temporal_angles"][t, j], transpose)
        return work if transpose else self._base(work, False)

    def b_rows(self, t, rows, inverse=False):
        p = self.arrays["block_permutation"]
        values = rows[:, p]
        first, second = values[:, ::2], values[:, 1::2]
        a, b, c = self.arrays["block_coefficients"][t].T
        if inverse:
            determinant = a * c - b * b
            a, b, c = c / determinant, -b / determinant, a / determinant
        result = np.empty_like(rows)
        result[:, p[::2]] = a * first + b * second
        result[:, p[1::2]] = b * first + c * second
        return result

    def factor_rows(self, t, rows):
        return self.b_rows(t, self.q_rows(t, rows, True) * np.sqrt(self.arrays["diagonal"][t])[None, :])

    def factor_transpose_rows(self, t, rows):
        return self.q_rows(t, self.b_rows(t, rows) * np.sqrt(self.arrays["diagonal"][t])[None, :])

    def hvp(self, t, vector):
        rows = np.asarray(vector, dtype=float)
        single = rows.ndim == 1
        rows = rows[None, :] if single else rows.T
        product = self.factor_transpose_rows(t, self.factor_rows(t, rows))
        return product[0] if single else product.T

    def score(self, t, prediction):
        white = self.factor_rows(t, (np.asarray(prediction) - self.reference(t))[None, :])[0]
        reference_white = self.factor_rows(t, self.reference(t)[None, :])[0]
        return {"energy_loss": float(white @ white / (reference_white @ reference_white))}

    def reference_factor_loss(self, t):
        white = self.factor_rows(t, self.reference(t)[None, :])[0] - self.arrays["whitened_rhs"][t]
        return float(white @ white / (self.arrays["whitened_rhs"][t] @ self.arrays["whitened_rhs"][t]))

    def independent_reference(self, t):
        white = self.arrays["whitened_rhs"][t][None, :]
        middle = self.b_rows(t, white, inverse=True) / np.sqrt(self.arrays["diagonal"][t])[None, :]
        return self.q_rows(t, middle)[0]


def normalized_gram_spectrum(gram, tails=(16, 48)):
    """Ranks of equal-norm columns in ONE fixed inner product.

    The supplied Gram is X.T M X for fixed SPD M, not a stack of unrelated
    time-dependent whitening frames. Eigenvalue threshold affects only the
    reported numerical-rank estimate; PR/entropy/tails use the clipped spectrum.
    """
    gram = finite(gram, "temporal Gram matrix")
    require(gram.ndim == 2 and gram.shape[0] == gram.shape[1], "Gram must be square")
    diagonal = np.diag(gram)
    require(np.all(diagonal > 0), "zero/nonpositive temporal column energy")
    scaling = np.sqrt(diagonal)
    normalized = gram / scaling[:, None] / scaling[None, :]
    normalized = (normalized + normalized.T) / 2
    eigenvalues = np.linalg.eigvalsh(normalized)[::-1]
    require(eigenvalues[-1] >= -1e-8 * max(1., eigenvalues[0]), "temporal Gram is not PSD")
    eigenvalues = np.maximum(eigenvalues, 0.)
    probability = eigenvalues / eigenvalues.sum()
    positive = probability > 0
    participation = float(1. / np.sum(probability ** 2))
    entropy = float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))
    return {"columns": gram.shape[0], "maximum_possible_rank": gram.shape[0],
            "participation_ratio": participation, "entropy_effective_rank": entropy,
            "stable_rank": float(eigenvalues.sum() / eigenvalues[0]),
            "numerical_rank": int(np.count_nonzero(eigenvalues > 1e-10 * eigenvalues[0])),
            "numerical_rank_relative_eigenvalue_threshold": 1e-10,
            "tail_energy_fraction": {str(k): float(probability[k:].sum()) for k in tails},
            "normalized_singular_values": np.sqrt(eigenvalues).tolist()}


def galerkin_projection(reference, history, hvp, *, reference_product=None, basis_rtol=1e-12):
    """Best current-H energy approximation in span(history).

    Rank-revealing SVD is performed AFTER normalizing the historical columns.
    This avoids unstable normal equations for nearly repeated solutions. No
    future reference is used. Returned error is formed explicitly rather than
    subtracting almost-equal scalar energies.
    """
    reference = finite(reference, "reference")
    href = finite(hvp(reference) if reference_product is None else reference_product, "H reference")
    denominator = float(reference @ href)
    require(denominator > 0, "nonpositive reference H energy")
    history = finite(history, "history")
    require(history.ndim == 2 and history.shape[0] == reference.size, "invalid history shape")
    if history.shape[1] == 0:
        return {"columns": 0, "basis_rank": 0, "relative_H_energy_error_squared": 1.,
                "relative_H_energy_error": 1., "coarse_orthogonality_relative": 0.}
    norms = np.linalg.norm(history, axis=0)
    history = history[:, norms > 0]
    norms = norms[norms > 0]
    if not norms.size:
        return {"columns": 0, "basis_rank": 0, "relative_H_energy_error_squared": 1.,
                "relative_H_energy_error": 1., "coarse_orthogonality_relative": 0.}
    left, singular, _ = np.linalg.svd(history / norms[None, :], full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * basis_rtol))
    basis = left[:, :rank]
    hbasis = finite(hvp(basis), "H history basis")
    coarse = basis.T @ hbasis
    coarse = (coarse + coarse.T) / 2
    require(np.linalg.eigvalsh(coarse)[0] > 0, "nonpositive Galerkin coarse operator")
    coefficient = np.linalg.solve(coarse, basis.T @ href)
    error = reference - basis @ coefficient
    herror = finite(hvp(error), "H projection error")
    energy = float(error @ herror)
    require(energy >= -1e-12 * denominator, "negative projected H energy")
    relative = max(0., energy) / denominator
    orthogonality = float(np.linalg.norm(basis.T @ herror) / max(np.linalg.norm(href), 1e-300))
    require(relative <= 1. + 1e-7, "Galerkin projection is worse than the zero point")
    return {"columns": int(norms.size), "basis_rank": rank,
            "relative_H_energy_error_squared": relative, "relative_H_energy_error": float(np.sqrt(relative)),
            "coarse_orthogonality_relative": orthogonality}


def diagnose_case(case, *, histories=(16, 48), probe_count=4, probe_seed=20261111,
                  reference_residual_tolerance=1e-7):
    begin = time.monotonic()
    n, stages = int(case.dimension), int(case.steps)
    require(n > 0 and stages > 1, "at least two nonempty systems are required")
    require(1 <= probe_count <= 64, "probe count must be in [1,64]")
    histories = tuple(sorted(set(int(k) for k in histories)))
    require(histories and all(k > 0 for k in histories), "history sizes must be positive")
    references = finite(np.stack([series_row(case, "reference", t) for t in range(stages)]), "reference trajectory")
    right_sides = finite(np.stack([series_row(case, "rhs", t) for t in range(stages)]), "RHS trajectory")
    require(references.shape == right_sides.shape == (stages, n), "trajectory shape differs from API")
    require(np.all(np.linalg.norm(right_sides, axis=1) > 0), "zero RHS outside relative-loss contract")
    require(np.all(np.linalg.norm(references, axis=1) > 0), "zero reference outside relative-loss contract")
    counters = {"requests": 0, "vectors": 0}

    def action(t, vectors):
        vectors = finite(vectors, "diagnostic probe")
        if vectors.ndim == 2 and vectors.shape[1] > 64:
            return np.column_stack([action(t, vectors[:, start:start + 64])
                                    for start in range(0, vectors.shape[1], 64)])
        value = finite(case.hvp(t, vectors), "diagnostic HVP")
        require(value.shape == vectors.shape, "HVP shape mismatch")
        counters["requests"] += 1
        counters["vectors"] += 1 if vectors.ndim == 1 else vectors.shape[1]
        return value

    # H_0 defines a COMMON metric for all columns. Time-varying Q_t alone is
    # never treated as a common whitening transformation.
    h0_references = action(0, references.T)
    euclidean_rank = normalized_gram_spectrum(references @ references.T, histories)
    h0_rank = normalized_gram_spectrum(references @ h0_references, histories)
    euclidean_rank["maximum_possible_rank"] = h0_rank["maximum_possible_rank"] = min(n, stages)
    rng = np.random.default_rng(probe_seed)
    probes = rng.choice([-1., 1.], size=(n, probe_count)) / np.sqrt(n)
    probe_products = [action(t, probes) for t in range(stages)]
    rows = []
    metadata = getattr(case, "meta", {}) or {}
    mu = getattr(case, "mu", metadata.get("mu"))
    upper = getattr(case, "L", metadata.get("L"))
    for t in range(stages):
        x, b = references[t], right_sides[t]
        hx = action(t, x)
        denominator = float(x @ hx)
        require(denominator > 0, "nonpositive reference energy")
        residual = float(np.linalg.norm(hx - b) / np.linalg.norm(b))
        require(residual <= reference_residual_tolerance, "reference residual exceeds audit tolerance")
        dot_difference = abs(float(x @ b) - denominator) / denominator
        sampled_gram = probes.T @ probe_products[t]
        symmetry = float(np.linalg.norm(sampled_gram - sampled_gram.T) /
                         max(np.linalg.norm(sampled_gram), 1e-300))
        require(symmetry < 1e-10, "sampled bilinear symmetry check fails")
        rayleigh = np.diag(sampled_gram)
        require(np.all(rayleigh > 0), "nonpositive sampled Rayleigh quotient")
        if mu is not None:
            require(np.min(rayleigh) >= float(mu) * (1 - 1e-8), "sample below supplied lower spectral bound")
        if upper is not None:
            require(np.max(rayleigh) <= float(upper) * (1 + 1e-8), "sample above supplied upper spectral bound")
        row = {"step": t, "reference_relative_residual": residual,
               "reference_energy_denominator_relative_disagreement": dot_difference,
               "sampled_bilinear_asymmetry": symmetry,
               "sampled_rayleigh_min": float(rayleigh.min()), "sampled_rayleigh_max": float(rayleigh.max()),
               "warm_start_relative_H_energy_error_squared": None,
               "rhs_adjacent_cosine": None, "solution_adjacent_cosine": None,
               "current_H_solution_adjacent_cosine": None,
               "relative_rhs_innovation": None, "relative_solution_innovation": None,
               "sampled_relative_operator_frobenius_drift": None,
               "sampled_relative_commutator": None, "history": {}}
        if t:
            previous = references[t - 1]
            delta = x - previous
            warm_energy = float(delta @ action(t, delta) / denominator)
            require(warm_energy >= -1e-12, "negative warm-start energy")
            hprevious = action(t, previous)
            previous_energy = float(previous @ hprevious)
            row.update(warm_start_relative_H_energy_error_squared=max(0., warm_energy),
                       rhs_adjacent_cosine=cosine(right_sides[t - 1], b),
                       solution_adjacent_cosine=cosine(previous, x),
                       current_H_solution_adjacent_cosine=float(np.clip(previous @ hx /
                           np.sqrt(previous_energy * denominator), -1., 1.)),
                       relative_rhs_innovation=float(np.linalg.norm(b - right_sides[t - 1]) / np.linalg.norm(b)),
                       relative_solution_innovation=float(np.linalg.norm(delta) / np.linalg.norm(x)))
            old, new = probe_products[t - 1], probe_products[t]
            row["sampled_relative_operator_frobenius_drift"] = float(np.linalg.norm(new - old) /
                max(np.linalg.norm(old), np.linalg.norm(new), 1e-300))
            left, right = action(t, old), action(t - 1, new)
            row["sampled_relative_commutator"] = float(np.linalg.norm(left - right) /
                max(np.linalg.norm(left) + np.linalg.norm(right), 1e-300))
        for size in histories:
            past = references[max(0, t - size):t].T
            row["history"][str(size)] = galerkin_projection(x, past, lambda v, t=t: action(t, v),
                                                            reference_product=hx)
        rows.append(row)
    history_summary = {}
    for size in histories:
        key = str(size)
        history_summary[key] = {
            "all_noninitial_steps": summarize([r["history"][key]["relative_H_energy_error_squared"] for r in rows[1:]]),
            "full_windows_only": summarize([r["history"][key]["relative_H_energy_error_squared"] for r in rows if r["step"] >= size]),
            "full_window_steps_zero_based": [r["step"] for r in rows if r["step"] >= size],
            "first_step_zero_subspace_error_squared": rows[0]["history"][key]["relative_H_energy_error_squared"]}
    names = ("warm_start_relative_H_energy_error_squared", "rhs_adjacent_cosine", "solution_adjacent_cosine",
             "current_H_solution_adjacent_cosine", "relative_rhs_innovation", "relative_solution_innovation",
             "sampled_relative_operator_frobenius_drift", "sampled_relative_commutator")
    return {"status": "diagnostics_computed", "dimension": n, "steps": stages,
            "metadata": {key: metadata[key] for key in ("family", "profile", "level", "schema") if key in metadata},
            "normalized_temporal_rank_euclidean": euclidean_rank,
            "normalized_temporal_rank_fixed_H0": h0_rank,
            "history_projection": history_summary,
            "adjacent_summary": {key: summarize([r[key] for r in rows]) for key in names},
            "max_reference_relative_residual": max(r["reference_relative_residual"] for r in rows),
            "max_reference_energy_denominator_relative_disagreement": max(r["reference_energy_denominator_relative_disagreement"] for r in rows),
            "diagnostic_oracle_usage": counters, "diagnostic_seconds": time.monotonic() - begin,
            "per_step": rows,
            "scope_limits": [
                "True-past-solution Galerkin projection is a diagnostic, not a candidate or an oracle lower bound for arbitrary solvers.",
                "No current/future true solution enters the history subspace. The current reference is used only to evaluate the oracle projection.",
                "H0 is one fixed common metric; per-stage whitened coordinates must not be stacked as if they were a fixed frame.",
                "With T=50, rank is at most 50 and history48 has only two fully populated windows (steps48,49).",
                "Frobenius drift, commutator and Rayleigh values are finite-probe diagnostics, not certified operator norms or spectral bounds.",
                "High rank or nonzero projection error alone does not prove a wall-clock-hard instance; actual independent solver screening is separate.",
                "All diagnostic work is outside solver timing, HVP and memory budgets."]}


def load_model(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("ihvp_hard_diagnostics_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--manifest", required=True, action="append", type=Path)
    parser.add_argument("--model", type=Path, default=root / "tasks/ihvp-sequence/versions/v4/curator/model.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--histories", type=int, nargs="+", default=[16, 48])
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--reference-rtol", type=float, default=1e-7)
    args = parser.parse_args()
    model = load_model(args.model)
    report = {"status": "PASS", "model_sha256": sha256(args.model), "audit_script_sha256": sha256(__file__),
              "scope": "private post-generation mathematical and temporal diagnostics; no hardness claim", "splits": [], "errors": []}
    for manifest_path in args.manifest:
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text())
        split = {"split": manifest.get("split"), "manifest_sha256": sha256(manifest_path), "cases": []}
        for entry in manifest["cases"]:
            try:
                path = (manifest_path.parent / entry["path"]).resolve()
                require(path.is_relative_to(manifest_path.parent), "case path escapes manifest root")
                digest = sha256(path)
                require(digest == entry["sha256"], "case SHA256 differs from frozen manifest")
                case = IndependentHardCase(path)
                diagnosis = diagnose_case(case, histories=args.histories, probe_count=args.probes,
                                         reference_residual_tolerance=args.reference_rtol)
                diagnosis["independent_factor_spectral_certificate"] = case.factor_spectral_certificate
                diagnosis["max_independent_reference_relative_l2_difference"] = max(
                    float(np.linalg.norm(case.reference(t) - case.independent_reference(t)) / np.linalg.norm(case.reference(t)))
                    for t in range(case.steps))
                diagnosis["max_reference_factor_energy_loss"] = max(case.reference_factor_loss(t)
                                                                       for t in range(case.steps))
                require(diagnosis["max_independent_reference_relative_l2_difference"] < 1e-9,
                        "supplied and independently reconstructed references differ")
                require(diagnosis["max_reference_factor_energy_loss"] < 1e-16,
                        "reference factor-energy error exceeds audit tolerance")
                teacher = model.HardCase(path)
                probes = np.random.default_rng(5831).standard_normal((case.dimension, 3))
                hvp_discrepancy = score_discrepancy = 0.
                for tick in sorted({0, case.steps // 2, case.steps - 1}):
                    expected = case.hvp(tick, probes)
                    actual = np.asarray(teacher.hvp(tick, probes))
                    discrepancy = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
                    require(discrepancy < 1e-10, "model HVP differs from independent factor implementation")
                    hvp_discrepancy = max(hvp_discrepancy, discrepancy)
                    prediction = .5 * case.reference(tick) + 1e-3 * probes[:, 0]
                    expected_loss = case.score(tick, prediction)["energy_loss"]
                    actual_loss = teacher.score(tick, prediction)["energy_loss"]
                    discrepancy = abs(actual_loss - expected_loss) / max(abs(expected_loss), 1e-300)
                    require(discrepancy < 1e-9, "model score differs from independent factor implementation")
                    score_discrepancy = max(score_discrepancy, discrepancy)
                diagnosis["max_backend_crosscheck_HVP_relative_difference"] = hvp_discrepancy
                diagnosis["max_backend_crosscheck_score_relative_difference"] = score_discrepancy
                diagnosis.update(case=entry["id"], data_sha256=digest)
                split["cases"].append(diagnosis)
                print(json.dumps({"split": split["split"], "case": entry["id"], "status": "diagnosed"}), flush=True)
            except Exception as error:
                report["status"] = "FAIL"
                report["errors"].append({"split": split["split"], "case": entry.get("id"),
                                         "error": f"{type(error).__name__}: {error}"})
        report["splits"].append(split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.out.resolve()), "errors": len(report["errors"])}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
